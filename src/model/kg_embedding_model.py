import torch
import torch.nn as nn
import torch.nn.functional as F


class TransH(nn.Module):
    def __init__(self, entity_num, relation_num, embed_size, margin=1., C=1., epsilon=1e-5, norm=1.,
                 entity_embedding=None):
        super(TransH, self).__init__()
        self.embed_size = embed_size
        self.margin = margin
        self.C = C
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.epsilon = epsilon
        self.norm = norm

        if entity_embedding is not None:
            self.entity_embedding = entity_embedding
        else:
            self.entity_embedding = torch.nn.Embedding(num_embeddings=entity_num,
                                                       embedding_dim=self.embed_size)
        self.relation_norm_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                          embedding_dim=self.embed_size)
        self.relation_hyper_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                           embedding_dim=self.embed_size)
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean")

        self.data_init()

    def data_init(self):
        nn.init.xavier_uniform_(self.relation_norm_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_hyper_embedding.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)

    def projected(self, ent, norm):
        """
        :param ent:  [B, D]
        :param norm: [B, D]
        :return:
        """
        norm = F.normalize(norm, p=2, dim=-1)

        return ent - torch.sum(ent * norm, dim=1, keepdim=True) * norm

    def distance(self, h, r, t):
        """
        h: [B]
        r: [B]
        t: [B]
        """
        head = self.entity_embedding(h)
        r_norm = self.relation_norm_embedding(r)
        r_hyper = self.relation_hyper_embedding(r)
        tail = self.entity_embedding(t)

        head_hyper = self.projected(head, r_norm)
        tail_hyper = self.projected(tail, r_norm)

        distance = head_hyper + r_hyper - tail_hyper
        score = torch.norm(distance, p=self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):

        head = self.entity_embedding(h.cuda())
        r_norm = self.relation_norm_embedding(r.cuda())
        r_hyper = self.relation_hyper_embedding(r.cuda())
        tail = self.entity_embedding(t.cuda())

        head_hyper = self.projected(head, r_norm)
        tail_hyper = self.projected(tail, r_norm)

        distance = head_hyper + r_hyper - tail_hyper
        score = torch.norm(distance, p=self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                ) - torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
            ))

    def orthogonal_loss(self, relation_embedding, w_embedding):
        dot = torch.sum(relation_embedding * w_embedding, dim=1, keepdim=False) ** 2
        norm = torch.norm(relation_embedding, p=self.norm, dim=1) ** 2

        loss = torch.sum(
            torch.relu(dot / norm - torch.autograd.Variable(torch.FloatTensor([self.epsilon]).cuda() ** 2)))
        return loss

    def forward(self, current_triples, corrupted_triples):
        """
        :param current_triples: [B, 3]
        :param corrupted_triples: [B, 3]
        :return:
        """
        h, r, t = torch.chunk(current_triples, 3, dim=1)  # [B, 1]
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

        h = torch.squeeze(h, dim=1).cuda()  # [B]
        r = torch.squeeze(r, dim=1).cuda()
        t = torch.squeeze(t, dim=1).cuda()
        h_c = torch.squeeze(h_c, dim=1).cuda()
        r_c = torch.squeeze(r_c, dim=1).cuda()
        t_c = torch.squeeze(t_c, dim=1).cuda()

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        y = -torch.ones_like(h, device=h.device)
        loss = self.loss_F(pos, neg, y)

        entity_embedding = self.entity_embedding(torch.cat([h, t, h_c, t_c]).cuda())
        relation_embedding = self.relation_hyper_embedding(torch.cat([r, r_c]).cuda())
        w_embedding = self.relation_norm_embedding(torch.cat([r, r_c]).cuda())

        orthogonal_loss = self.orthogonal_loss(relation_embedding, w_embedding)

        scale_loss = self.scale_loss(entity_embedding)

        return loss + self.C * (scale_loss / len(entity_embedding) + orthogonal_loss / len(relation_embedding))
