import torch 
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    '''
    Knowledge Distillation Loss
    '''
    def __init__(self, temperature=4, alpha=.7, pos_weigths = None):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weigths)
        self.bce = BinaryFocalLoss(alpha=0.7)
        #self.embedding_loss = nn.CosineEmbeddingLoss()
        self.embedding_loss = nn.KLDivLoss(reduction="none")
        self.epsilon = 1e-3

        

    def forward(self, student_logits, teacher_logits, labels, teacher_features, student_features):
  
            ## -----------
            # soft features
            ## ------------
            soft_teacher = nn.functional.softmax(teacher_features, dim=-1)
            soft_prob = nn.functional.log_softmax(student_features, dim=-1)


            ##---------
            # calculating weights
            ##--------- 
            with torch.no_grad():
                teacher_pred = (torch.sigmoid(teacher_logits) > 0.5).int().squeeze(1)
                correct_mask = (teacher_pred ==labels).float()
                weights = 5 * correct_mask + (1/5)* (1 - correct_mask)

            ##-------------
            ## Feature loss
            ##-------------
            embedding_loss = self.embedding_loss(soft_prob, soft_teacher)
            #cosine_loss = (weights.view(-1, 1,1) * embedding_loss).mean() 
            cosine_loss = embedding_loss.mean()



            ##-------------
            # classification loss
            ##-------------
            label_loss = self.bce(student_logits.squeeze(1), labels.float())
                        ##------------
            # Gradient normalization 
            ##------------
            kd_grad = torch.autograd.grad(
                cosine_loss, student_features, 
                retain_graph=True, create_graph=True
            )[0].norm()
            
            ce_grad = torch.autograd.grad(
                label_loss, student_logits, 
                retain_graph=True, create_graph=True
            )[0].norm()
            
            kd_weight = (ce_grad/ (kd_grad+ 1e-6)).detach()
            ##-----------
            # total loss
            ##-----------
            loss = ((1-self.alpha)*kd_weight * (1/self.temperature**2)*cosine_loss) + self.alpha * label_loss 

            return loss , kd_weight*cosine_loss, label_loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor for positive class (float)
        :param gamma: Focusing parameter for modulating factor (1 - p_t)
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: raw predictions (before sigmoid), shape [batch_size]
        :param targets: binary ground truth (0 or 1), shape [batch_size]
        """
        prob = torch.sigmoid(logits)
        targets = targets.float()
        
        pt = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
if __name__ == "__main__":
     loss = DistillationLoss()
     teacher_logits = torch.rand(size = (1, 2))
     student_logits = torch.rand(size = (1, 2))
     labels = torch.rand(size=(1,1))
     teacher_features = torch.rand(size= (1, 32, 16))
     student_features = torch.rand(size = (1, 32, 128))
     target = torch.ones(1)
     loss.forward(student_logits, teacher_logits, labels, teacher_features, student_features, target)


