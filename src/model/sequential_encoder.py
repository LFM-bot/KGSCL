import torch.nn as nn
import torch
import copy
import math
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_blocks, num_heads, attn_dropout, hidden_dropout,
                 layer_norm_eps=0.02, bidirectional=False):
        super(Transformer, self).__init__()
        self.bidirectional = bidirectional
        encoder_layer = EncoderLayer(embed_size=embed_size,
                                     ffn_hidden=ffn_hidden,
                                     num_heads=num_heads,
                                     attn_dropout=attn_dropout,
                                     hidden_dropout=hidden_dropout,
                                     layer_norm_eps=layer_norm_eps)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_blocks)])

    def forward(self, item_input, seq_embedding):
        """
        Only output the sequence representations of the last layer in Transformer.
        out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        mask = self.create_mask(item_input)
        for layer in self.encoder_layers:
            seq_embedding = layer(seq_embedding, mask)
        return seq_embedding

    def create_mask(self, input_seq):
        """
        Parameters:
            input_seq: torch.LongTensor, [batch_size, max_len]
        Return:
            mask: torch.BoolTensor, [batch_size, 1, max_len, max_len]
        """
        mask = (input_seq != 0).bool().unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_len]
        mask = mask.expand(-1, -1, mask.size(-1), -1)
        if not self.bidirectional:
            mask = torch.tril(mask)
        return mask

    def set_attention_direction(self, bidirection=False):
        self.bidirectional = bidirection


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_heads, attn_dropout, hidden_dropout, layer_norm_eps):
        super(EncoderLayer, self).__init__()

        self.attn_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.pff_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)

        self.self_attention = MultiHeadAttentionLayer(embed_size, num_heads, attn_dropout)
        self.pff = PointWiseFeedForwardLayer(embed_size, ffn_hidden)

        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.pff_out_drop = nn.Dropout(hidden_dropout)

    def forward(self, input_seq, inputs_mask):
        """
        input:
            inputs: torch.FloatTensor, [batch_size, max_len, embed_size]
            inputs_mask: torch.BoolTensor, [batch_size, 1, 1, max_len]
        return:
            out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        out_seq, att_matrix = self.self_attention(input_seq, input_seq, input_seq, inputs_mask)
        input_seq = self.attn_layer_norm(input_seq + self.hidden_dropout(out_seq))
        out_seq = self.pff(input_seq)
        out_seq = self.pff_layer_norm(input_seq + self.pff_out_drop(out_seq))
        return out_seq


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, nhead, attn_dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.nhead = nhead

        if self.embed_size % self.nhead != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.embed_size, self.nhead)
            )
        self.head_dim = self.embed_size // self.nhead

        # Q K V input linear layer
        self.fc_q = nn.Linear(self.embed_size, self.embed_size)
        self.fc_k = nn.Linear(self.embed_size, self.embed_size)
        self.fc_v = nn.Linear(self.embed_size, self.embed_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc_o = nn.Linear(self.embed_size, self.embed_size)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim).float()))

    def forward(self, query, key, value, inputs_mask=None):
        """
        :param query: [query_size, max_len, embed_size]
        :param key: [key_size, max_len, embed_size]
        :param value: [key_size, max_len, embed_size]
        :param inputs_mask: [N, 1, max_len, max_len]
        :return: [N, max_len, embed_size]
        """
        batch_size = query.size(0)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # [batch_size, n_head, max_len, head_dim]
        Q = Q.view(query.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        K = K.view(key.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        V = V.view(value.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))

        # calculate attention score
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if inputs_mask is not None:
            energy = energy.masked_fill(inputs_mask == 0, -1.e10)

        attention_prob = F.softmax(energy, dim=-1)
        attention_prob = self.attn_dropout(attention_prob)

        out = torch.matmul(attention_prob, V)  # [batch_size, n_head, max_len, head_dim]
        out = out.permute((0, 2, 1, 3)).contiguous()  # memory layout
        out = out.view((batch_size, -1, self.embed_size))
        out = self.fc_o(out)
        return out, attention_prob


class PointWiseFeedForwardLayer(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(PointWiseFeedForwardLayer, self).__init__()

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, inputs):
        out = self.fc2(F.gelu(self.fc1(inputs)))
        return out


class LightTransformerEncoder(nn.Module):
    r""" One LightTransformerEncoder consists of several LightTransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'.
            candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(self,
                 n_layers=2,
                 n_heads=2,
                 k_interests=5,
                 hidden_size=64,
                 seq_len=50,
                 inner_size=256,
                 hidden_dropout_prob=0.5,
                 attn_dropout_prob=0.5,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):

        super(LightTransformerEncoder, self).__init__()
        layer = LightTransformerLayer(n_heads, k_interests, hidden_size, seq_len, inner_size,
                                      hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LightTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """

    def __init__(self, n_heads, k_interests, hidden_size, seq_len, intermediate_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(LightTransformerLayer, self).__init__()
        self.multi_head_attention = LightMultiHeadAttention(n_heads, k_interests, hidden_size,
                                                            seq_len, hidden_dropout_prob, attn_dropout_prob,
                                                            layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size,
                                        hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, k_interests, hidden_size, seq_len, hidden_dropout_prob, attn_dropout_prob,
                 layer_norm_eps):
        super(LightMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = ItemToInterestAggregation(seq_len, hidden_size, k_interests)
        self.attpooling_value = ItemToInterestAggregation(seq_len, hidden_size, k_interests)

        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        # linear map
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(self.attpooling_value(mixed_value_layer))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = F.softmax(attention_scores, -1)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        # pos_query_layer = self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        pos_query_layer = self.transpose_for_scores(self.pos_q_linear(pos_emb))
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        # abs_pos_bias = nn.Softmax(dim=-1)(abs_pos_bias)
        abs_pos_bias = F.softmax(abs_pos_bias, -1)

        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ItemToInterestAggregation(nn.Module):
    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests  # k latent interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        D_matrix = torch.matmul(input_tensor, self.theta)  # [B, L, k]
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum('nij, nik -> nkj', input_tensor, D_matrix)  # #[B, k, d]

        return result


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
