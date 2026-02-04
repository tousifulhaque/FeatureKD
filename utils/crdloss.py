import torch 
import torch.nn.functional as F
import torch.nn as nn

TIMESTAMPS = 128
eps = 1e-7

class InfoNCELoss(nn.Module):
    '''
    InfoNCE Contrastive Loss - More effective than critic-based loss
    Used in SimCLR and MoCo for self-supervised learning
    
    Loss = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )
    '''
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, teacher_features, student_features):
        """
        Compute InfoNCE loss with in-batch negatives
        
        Args:
            teacher_features: Teacher representations (batch_size, feature_dim)
            student_features: Student representations (batch_size, feature_dim)
            
        Returns:
            loss: Scalar loss value
        """
        batch_size = teacher_features.size(0)
        
        # Normalize features
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        student_norm = F.normalize(student_features, p=2, dim=-1)
        
        # Compute similarity matrix (batch_size x batch_size)
        # similarity[i,j] = cosine_similarity(teacher[i], student[j])
        similarity_matrix = torch.matmul(teacher_norm, student_norm.T) / self.temperature
        
        # Create labels - diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=teacher_features.device)
        
        # Compute cross-entropy loss
        # Each row should predict its corresponding column (diagonal)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class CriticContrastiveLoss(nn.Module):
    '''
    Critic-based Contrastive Loss for Mutual Information Estimation
    Based on: f_S* = arg max_{f_S} max_h L_critic(h)
    where L_critic(h) = E[log h(T,S)|C=1] + N * E[log(1-h(T,S))|C=0]
    and h(T,S) = exp(g_T(T)' g_S(S) / τ) / (exp(g_T(T)' g_S(S) / τ) + N/M)
    '''
    def __init__(self, temperature=0.07, N=1.0, M=1.0, eps=1e-8):
        """
        Args:
            temperature (float): Temperature parameter τ for scaling
            N (float): Number of negative samples weight
            M (float): Normalization constant
            eps (float): Small constant for numerical stability
        """
        super(CriticContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.N = N
        self.M = M
        self.eps = eps
        
    def compute_critic(self, teacher_features, student_features):
        """
        Compute critic function h(T, S)
        h(T,S) = exp(g_T(T)' g_S(S) / τ) / (exp(g_T(T)' g_S(S) / τ) + N/M)
        
        Args:
            teacher_features: Teacher representations (batch_size, feature_dim)
            student_features: Student representations (batch_size, feature_dim)
            
        Returns:
            h: Critic values (batch_size,)
        """
        # Normalize features to unit vectors
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        student_norm = F.normalize(student_features, p=2, dim=-1)
        
        # Compute dot product similarity
        similarity = torch.sum(teacher_norm * student_norm, dim=-1)  # (batch_size,)
        
        # Compute scaled exponential
        exp_sim = torch.exp(similarity / self.temperature)
        
        # Compute critic function h(T, S)
        h = exp_sim / (exp_sim + self.N / self.M + self.eps)
        
        return h
    
    def forward(self, teacher_features, student_features, is_positive):
        """
        Compute the critic contrastive loss
        
        Args:
            teacher_features: Teacher representations (batch_size, feature_dim)
            student_features: Student representations (batch_size, feature_dim)
            is_positive: Binary labels indicating positive (1) or negative (0) pairs (batch_size,)
            
        Returns:
            loss: Scalar loss value (negative of L_critic for minimization)
        """
        # Get device from teacher features
        device = teacher_features.device
        
        # Move all tensors to the same device
        student_features = student_features.to(device)
        is_positive = is_positive.to(device)
        
        # Flatten features if needed
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.view(teacher_features.size(0), -1)
        if student_features.dim() > 2:
            student_features = student_features.view(student_features.size(0), -1)
        
        # Compute critic values
        h = self.compute_critic(teacher_features, student_features)
        
        # Clamp to avoid log(0)
        h = torch.clamp(h, min=self.eps, max=1.0 - self.eps)
        
        # Convert is_positive to float if needed
        if is_positive.dtype != torch.float:
            is_positive = is_positive.float()
        
        # Compute positive and negative terms
        # For positive pairs (C=1): log h(T,S)
        positive_term = is_positive * torch.log(h)
        
        # For negative pairs (C=0): N * log(1 - h(T,S))
        negative_term = (1 - is_positive) * self.N * torch.log(1 - h)
        
        # Total loss (we want to maximize, so negate for minimization)
        L_critic = torch.mean(positive_term + negative_term)
        
        # Return negative for gradient descent (we want to maximize L_critic)
        loss = -L_critic
        
        return loss


class CRDContrastLoss(nn.Module):
    """
    CRD-style contrastive loss without memory bank
    Uses create_negative_samples for negative sampling
    """
    def __init__(self, temperature=0.07, n_negatives=None):
        super(CRDContrastLoss, self).__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        
    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: anchor features (batch_size, feat_dim) - normalized
            positive: positive features (batch_size, feat_dim) - normalized  
            negatives: negative features (batch_size, n_neg, feat_dim) - normalized
            
        Returns:
            Contrastive loss following CRD formulation
        """
        batch_size = anchor.size(0)
        
        # Compute positive similarity (batch_size,)
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        pos_prob = torch.exp(pos_sim)
        
        # Compute negative similarities (batch_size, n_neg)
        if negatives.dim() == 2:
            # If negatives is (batch_size, feat_dim), expand to (batch_size, 1, feat_dim)
            negatives = negatives.unsqueeze(1)
        
        # anchor: (batch_size, feat_dim) -> (batch_size, 1, feat_dim)
        # negatives: (batch_size, n_neg, feat_dim)
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature  # (batch_size, n_neg)
        neg_prob = torch.exp(neg_sim)
        
        # CRD loss formulation: -log(pos / (pos + sum(neg)))
        # Equivalent to NCE loss with noise distribution
        denom = pos_prob + neg_prob.sum(1) + eps
        loss = -torch.log(pos_prob / denom).mean()
        
        return loss


class CCLWithProjection(nn.Module):
    '''
    CRD-style Contrastive Loss with learnable projection heads
    Symmetric loss: both teacher→student and student→teacher
    '''
    def __init__(self, teacher_dim, student_dim, projection_dim=256, 
                 temperature=0.07, n_negatives=63, eps=1e-8,
                 device='cpu'):
        """
        Args:
            teacher_dim (int): Dimension of teacher features
            student_dim (int): Dimension of student features
            projection_dim (int): Dimension of projection space
            temperature (float): Temperature parameter for contrastive loss
            n_negatives (int): Number of negative samples per positive
            eps (float): Small constant for numerical stability
        """
        super(CCLWithProjection, self).__init__()
        
        self.temperature = temperature
        self.n_negatives = n_negatives
        
        # Embedding/Projection heads (like CRD's Embed)
        hidden_dim = projection_dim * 2
        
        self.embed_t = nn.Sequential(
            nn.Linear(teacher_dim*TIMESTAMPS, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        ).to(device)
        
        self.embed_s = nn.Sequential(
            nn.Linear(student_dim*TIMESTAMPS, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        ).to(device)
        
        # CRD-style contrastive losses (symmetric)
        self.criterion_t = CRDContrastLoss(temperature=temperature, n_negatives=n_negatives)
        self.criterion_s = CRDContrastLoss(temperature=temperature, n_negatives=n_negatives)

    def create_negative_samples(self, features, num_negatives=None):
        '''
        Create negative samples for CRD-style contrastive learning
        Returns multiple negatives per sample
        
        :param features: Input features (batch_size, feat_dim)
        :param num_negatives: Number of negatives per sample (default: batch_size - 1)
        :return: Negative samples (batch_size, n_neg, feat_dim)
        '''
        batch_size = features.size(0)
        feat_dim = features.size(1)
        
        if num_negatives is None:
            num_negatives = batch_size - 1
        
        # Create negative samples by sampling from other elements in batch
        negatives = []
        for i in range(batch_size):
            # Get all indices except current one
            neg_indices = torch.cat([torch.arange(i, device=features.device), 
                                     torch.arange(i+1, batch_size, device=features.device)])
            
            # Sample num_negatives random negatives
            if len(neg_indices) >= num_negatives:
                sampled_idx = neg_indices[torch.randperm(len(neg_indices), device=features.device)[:num_negatives]]
            else:
                # If not enough negatives, sample with replacement
                sampled_idx = neg_indices[torch.randint(0, len(neg_indices), (num_negatives,), device=features.device)]
            
            negatives.append(features[sampled_idx])
        
        # Stack to (batch_size, n_neg, feat_dim)
        negatives = torch.stack(negatives, dim=0)
        return negatives
    
    def forward(self, teacher_features, student_features, is_positive=None):
        """
        CRD-style symmetric contrastive loss
        
        Args:
            teacher_features: Teacher representations (batch_size, teacher_dim)
            student_features: Student representations (batch_size, student_dim)
            is_positive: Not used (kept for compatibility)
            
        Returns:
            loss: Scalar loss value (sum of teacher→student and student→teacher losses)
        """
        # Get device from projection head
        device = next(self.embed_t.parameters()).device
        
        # Move all tensors to the same device
        teacher_features = teacher_features.to(device)
        student_features = student_features.to(device)
        
        # Flatten features if needed
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.view(teacher_features.size(0), -1)
        if student_features.dim() > 2:
            student_features = student_features.view(student_features.size(0), -1)
        
        # Embed/project features
        f_t = self.embed_t(teacher_features)
        f_s = self.embed_s(student_features)
        
        # L2 normalize (like CRD)
        f_t = F.normalize(f_t, p=2, dim=-1)
        f_s = F.normalize(f_s, p=2, dim=-1)
        
        # Create negative samples
        neg_t = self.create_negative_samples(f_t, self.n_negatives)  # (batch, n_neg, feat_dim)
        neg_s = self.create_negative_samples(f_s, self.n_negatives)  # (batch, n_neg, feat_dim)
        
        # Symmetric losses like CRD
        # (a) teacher as anchor, student as positive, sample negatives from student
        loss_t = self.criterion_t(f_t, f_s, neg_s)
        
        # (b) student as anchor, teacher as positive, sample negatives from teacher  
        loss_s = self.criterion_s(f_s, f_t, neg_t)
        
        # Total loss
        loss = loss_t + loss_s
        
        return loss
    
class CCKDLoss(nn.Module): 
    '''
    Knowledge Distillation Loss with CRD-style contrastive learning
    Symmetric contrastive loss + classification loss
    '''
    def __init__(self, teacher_dim, student_dim, projection_dim=256, 
                 temperature=0.07, batch_size=64, eps=1e-8, device='cpu'):
        super(CCKDLoss, self).__init__()
        self.temperature = temperature
        self.label_loss_func = nn.BCEWithLogitsLoss()
        
        # CRD-style contrastive loss with symmetric formulation
        # n_negatives = batch_size - 1 (all other samples in batch)
        self.ccl = CCLWithProjection(
            teacher_dim=teacher_dim, 
            student_dim=student_dim,
            projection_dim=projection_dim, 
            temperature=temperature, 
            n_negatives=batch_size-1,
            eps=eps, 
            device=device
        )
        
        
    def forward(self, x_s, x_t, pred_s, pred_t, gt): 
        '''
        Docstring for forward
        
        :param self: Description
        :param x_s: teacher features
        :param x_t: student features
        :param pred_s: student 
        ''' 
        # Get device from student features
        device = x_s.device
        
        # Move all tensors to the same device
        x_t = x_t.to(device)
        pred_s = pred_s.to(device)
        pred_t = pred_t.to(device)
        gt = gt.to(device)
        
        # create shuffled negative samples  
        batch_size = x_s.size(0)
        if x_s.dim() > 2: 
            x_s = x_s.reshape(batch_size, -1)

        if x_t.dim() > 2 : 
            x_t = x_t.reshape(batch_size, -1)

        # CRD-style symmetric contrastive loss
        # No need to manually create negatives - handled inside CCLWithProjection
        cl_loss = self.ccl(x_t, x_s)  # Symmetric: teacher↔student
        
        # Classification loss
        class_loss = self.label_loss_func(pred_s.squeeze(1), gt)
        
        # Combined loss
        total_loss =  .3 * (self.temperature**2) * cl_loss + .7 * class_loss  
        return total_loss, cl_loss, class_loss



            
            
        

if __name__ == "__main__":
    print("="*80)
    print("Testing Loss Functions")
    print("="*80)
    
    # Test 1: CriticContrastiveLoss
    print("\n[Test 1] CriticContrastiveLoss")
    print("-" * 50)
    critic_loss = CriticContrastiveLoss(temperature=0.07, N=1.0, M=1.0)
    
    # Test with 2D features
    batch_size = 8
    feature_dim = 128
    teacher_feat = torch.rand(batch_size, feature_dim)
    student_feat = torch.rand(batch_size, feature_dim)
    is_positive = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    
    loss_val = critic_loss(teacher_feat, student_feat, is_positive)
    print(f"✓ Input shapes: teacher={teacher_feat.shape}, student={student_feat.shape}")
    print(f"✓ Positive/Negative pairs: {is_positive.sum().int()}/{(1-is_positive).sum().int()}")
    print(f"✓ Loss value: {loss_val.item():.6f}")
    
    # Test with 3D features (should flatten automatically)
    teacher_feat_3d = torch.rand(batch_size, 32, 16)
    student_feat_3d = torch.rand(batch_size, 32, 16)
    loss_val_3d = critic_loss(teacher_feat_3d, student_feat_3d, is_positive)
    print(f"✓ 3D Input test: {loss_val_3d.item():.6f}")
    
    # Test 2: CCLWithProjection
    print("\n[Test 2] CCLWithProjection")
    print("-" * 50)
    teacher_dim = 128
    student_dim = 128
    projection_dim = 64
    
    ccl_proj = CCLWithProjection(
        teacher_dim=teacher_dim, 
        student_dim=student_dim, 
        projection_dim=projection_dim,
        temperature=0.07, 
        N=1.0, 
        M=1.0
    )
    
    # Create features with temporal dimension
    teacher_feat = torch.rand(batch_size, teacher_dim * TIMESTAMPS)
    student_feat = torch.rand(batch_size, student_dim * TIMESTAMPS)
    is_positive = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    
    loss_proj_val = ccl_proj(teacher_feat, student_feat, is_positive)
    print(f"✓ Input shapes: teacher={teacher_feat.shape}, student={student_feat.shape}")
    print(f"✓ Projection dim: {projection_dim}")
    print(f"✓ Loss value: {loss_proj_val.item():.6f}")
    
    # Test negative sample creation
    print(f"✓ Testing negative sample creation...")
    neg_samples = ccl_proj.create_negative_samples(student_feat)
    print(f"✓ Negative samples shape: {neg_samples.shape}")
    print(f"✓ Negative samples are different from original: {not torch.equal(neg_samples, student_feat)}")
    
    # Test 3: CCKDLoss (Complete Knowledge Distillation Loss)
    print("\n[Test 3] CCKDLoss (Complete KD with Contrastive Loss)")
    print("-" * 50)
    batch_size = 16
    teacher_dim = 64
    student_dim = 64
    num_classes = 2
    
    cckd_loss = CCKDLoss(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        projection_dim=64,
        temperature=0.07,
        batch_size=batch_size
    )
    
    # Create sample inputs
    student_features = torch.rand(batch_size, student_dim * TIMESTAMPS)
    teacher_features = torch.rand(batch_size, teacher_dim * TIMESTAMPS)
    student_pred = torch.rand(batch_size, num_classes)  # Logits
    teacher_pred = torch.rand(batch_size, num_classes)  # Logits
    ground_truth = torch.randint(0, 2, (batch_size, num_classes)).float()  # Binary labels
    
    total_loss = cckd_loss(student_features, teacher_features, student_pred, teacher_pred, ground_truth)
    print(f"✓ Input shapes:")
    print(f"  - Student features: {student_features.shape}")
    print(f"  - Teacher features: {teacher_features.shape}")
    print(f"  - Student predictions: {student_pred.shape}")
    print(f"  - Ground truth: {ground_truth.shape}")
    print(f"✓ Total loss (CL + Classification): {total_loss.item():.6f}")
    
    # Test backward pass
    print(f"✓ Testing backward pass...")
    total_loss.backward()
    print(f"✓ Gradients computed successfully!")
    
    # Test 4: Edge Cases
    print("\n[Test 4] Edge Cases")
    print("-" * 50)
    
    # All positive pairs
    critic_loss_all_pos = CriticContrastiveLoss()
    is_positive_all = torch.ones(batch_size, dtype=torch.float32)
    loss_all_pos = critic_loss_all_pos(teacher_features, student_features, is_positive_all)
    print(f"✓ All positive pairs: {loss_all_pos.item():.6f}")
    
    # All negative pairs
    is_negative_all = torch.zeros(batch_size, dtype=torch.float32)
    loss_all_neg = critic_loss_all_pos(teacher_features, student_features, is_negative_all)
    print(f"✓ All negative pairs: {loss_all_neg.item():.6f}")
    
    # Single sample
    single_teacher = torch.rand(1, feature_dim)
    single_student = torch.rand(1, feature_dim)
    single_label = torch.tensor([1.0])
    loss_single = critic_loss_all_pos(single_teacher, single_student, single_label)
    print(f"✓ Single sample: {loss_single.item():.6f}")
    
    # Test 5: Gradient Flow Verification
    print("\n[Test 5] Gradient Flow Verification")
    print("-" * 50)
    
    # Create model with parameters
    test_teacher = nn.Linear(feature_dim, feature_dim)
    test_student = nn.Linear(feature_dim, feature_dim)
    
    # Forward pass
    teacher_input = torch.rand(batch_size, feature_dim, requires_grad=True)
    student_input = torch.rand(batch_size, feature_dim, requires_grad=True)
    
    teacher_out = test_teacher(teacher_input)
    student_out = test_student(student_input)
    
    is_positive = torch.cat([torch.ones(batch_size//2), torch.zeros(batch_size//2)])
    loss = critic_loss(teacher_out, student_out, is_positive)
    
    # Backward pass
    loss.backward()
    
    print(f"✓ Loss: {loss.item():.6f}")
    print(f"✓ Teacher input grad exists: {teacher_input.grad is not None}")
    print(f"✓ Student input grad exists: {student_input.grad is not None}")
    print(f"✓ Teacher weight grad norm: {test_teacher.weight.grad.norm().item():.6f}")
    print(f"✓ Student weight grad norm: {test_student.weight.grad.norm().item():.6f}")
    
    print("\n" + "="*80)
    print("All tests completed successfully! ✓")
    print("="*80)


