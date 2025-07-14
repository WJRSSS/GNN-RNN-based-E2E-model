import torch
import torch.nn as nn
from TwoLayerFC import TwoLayerFC
import numpy as np
from projection import *

class Student_Model(nn.Module):
    """
    This class holds the Neural Network for student model
    """

    def __init__(self,
                 hidden_size: int,
                 lr: float,
                 batch_size: int,
                 num_epochs: int,
                 covariate_size: int,
                 target_width: int,
                 device):
        #print(f"device is: {device}")
        super(Student_Model, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.covariate_size = covariate_size
        self.target_width = target_width
        
        self.TwoLayerFC = TwoLayerFC(input_size=covariate_size,
                                     hidden_size=hidden_size,
                                     output_size = target_width * (target_width - 1),
                                     device=device)
        self.TwoLayerFC.double()



    def train(self, A_tensor, B_tensor, train_target_df, train_covariate_df, teacher_df):
        device = self.device
        hidden_size = self.hidden_size
        lr = self.lr
        batch_size = self.batch_size
        num_epochs = self.num_epochs
        covariate_size = self.covariate_size
        target_width = self.target_width
        TwoLayerFC = self.TwoLayerFC
        # 定义优化器
        TwoLayerFC_optimizer = torch.optim.Adam(TwoLayerFC.parameters(), lr=lr)
        # 定义学习率调度器
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(TwoLayerFC_optimizer, 25, last_epoch=-1)

        train_target_df = np.array(train_target_df)
        train_covariate_df = np.array(train_covariate_df)
        teacher_df = np.array(teacher_df)

        train_target_tensor = torch.tensor(train_target_df)
        train_covariate_tensor = torch.tensor(train_covariate_df)
        teacher_tensor = torch.tensor(teacher_df)

        train_target_tensor = train_target_tensor.double()
        train_covariate_tensor = train_covariate_tensor.double()
        teacher_tensor = teacher_tensor.double()

        #TwoLayerFC.to(device)
        train_target_tensor.to(device)
        train_covariate_tensor.to(device)

        print(next(TwoLayerFC.parameters()).device)
        print(train_covariate_tensor.device)
        print(train_target_tensor.device)

        MSE_loss = nn.MSELoss()
        # λ:蒸馏温度：衡量老师与学生模型之间的差异
        lambda_1 = 0.2

        for i in range(num_epochs):
            print(f"epoch_num:{i}")
            epoch_loss_sum = 0.0
            TwoLayerFC_optimizer.zero_grad()
            outputs = TwoLayerFC(train_covariate_tensor)
            #outputs = projection_k(A_tensor,B_tensor, outputs)
            print(outputs.size())
            loss = MSE_loss(outputs, train_target_tensor)
            loss += lambda_1 * MSE_loss(outputs, teacher_tensor)
            loss.backward()
            TwoLayerFC_optimizer.step()
            scheduler1.step()
            print(f'Epoch [{i+1}], Loss: {loss.item():.4f}')


    
    def predict(self, A_tensor,B_tensor,test_target_df, test_covariate_df):
        device = self.device
        
        # Convert data to tensors
        test_covariate_tensor = torch.tensor(np.array(test_covariate_df), dtype=torch.double)
        test_target_tensor = torch.tensor(np.array(test_target_df), dtype=torch.double)
        
        # Move tensors to device
        #test_covariate_tensor = test_covariate_tensor.to(device)
        #test_target_tensor = test_target_tensor.to(device)
        
        # Set model to evaluation mode
        self.TwoLayerFC.eval()
        
        with torch.no_grad():  # Disable gradient calculation
            # Make predictions
            predictions = self.TwoLayerFC(test_covariate_tensor)
            predictions = projection_k(A_tensor, B_tensor, predictions)
            
            # Calculate test loss
            test_loss = nn.MSELoss()(predictions, test_target_tensor)
        
        # Convert predictions to numpy array
        predictions = predictions.cpu().numpy()
        
        print(f"Test MSE Loss: {test_loss.item():.4f}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {test_target_tensor.shape}")
        
        return predictions, test_loss.item()
