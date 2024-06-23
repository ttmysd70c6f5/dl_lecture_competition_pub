import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class DefossezClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 320,
        n_subject: int = 4,
    ) -> None:
        super().__init__()

        self.pre_conv_block = nn.Sequential(
            SimpleConvBlock(in_dim=in_channels, out_dim=in_channels, kernel_size=1),
        )
        self.subject_block = SubjectBlock(in_channels,in_channels,n_subject)
        self.post_conv_block = nn.Sequential(
            ConvBlock(in_dim=in_channels,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=0),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=1),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=2),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=3),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=4),
            SimpleConvBlock(in_dim = hid_dim, out_dim = hid_dim*2, kernel_size=1, activate = True),
            SimpleConvBlock(in_dim = hid_dim*2, out_dim = hid_dim*2, kernel_size=1, activate = True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim*2, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
            subject_idxs ( b ): subject index
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.pre_conv_block(X)
        X = self.subject_block(X, subject_idxs)
        X = self.post_conv_block(X)

        return self.head(X)

class Classifier2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 320,
        n_subject: int = 4,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(in_dim=in_channels,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=0),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=1),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=2),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=3),
            ConvBlock(in_dim=hid_dim,hid_dim=hid_dim,out_dim=hid_dim*2,kernel_size=3,k=4),
            SimpleConvBlock(in_dim = hid_dim, out_dim = hid_dim*2, kernel_size=1, activate = True),
            SimpleConvBlock(in_dim = hid_dim*2, out_dim = hid_dim*2, kernel_size=1, activate = True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim*2, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
            subject_idxs ( b ): subject index
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.block(X)

        return self.head(X)

class SimpleConvBlock(nn.Module):
    """Convlution block before/after subject block"""
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 1,
        p_drop: float = 0.1,
        activate: bool = False,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activate = activate

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv0(X)
        if self.activate:
            X = F.gelu(X)
        return self.dropout(X)

class ConvBlock(nn.Module):
    """Convolution layers after subject block"""
    def __init__(
        self,
        in_dim = 271,
        hid_dim = 320,
        out_dim = 640,
        kernel_size: int = 3,
        k: int = 0,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, hid_dim, kernel_size, padding="same",dilation=2**(2*k)%5)
        self.conv1 = nn.Conv1d(hid_dim, hid_dim, kernel_size, padding="same",dilation=2**(2*k+1)%5)
        self.conv2 = nn.Conv1d(hid_dim, out_dim, kernel_size, padding="same",dilation=2)
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=hid_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hid_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.hid_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        
        X = self.conv2(X)
        X = F.glu(X, dim=-2) # No normalization for 2nd convolution

        return self.dropout(X)
    
class SubjectBlock(nn.Module):
    """Subject linear layer"""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels)) # [S,C_in,C_out]
        self.weights.data *= 1 / in_channels**0.5 # Xavier initialization
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X:torch.Tensor, subject_idxs: torch.Tensor):
        _, C_in, C_out = self.weights.shape
        weights = self.weights.gather(0, subject_idxs.view(-1, 1, 1).expand(-1, C_in, C_out)) # Assign subject-specific weights: [B,C_in,C_out]
        X = torch.einsum("bct,bcd->bdt", X, weights)
        X = self.batchnorm(X)
        return X
    
    
    
    
class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        n_subject: int = 4,
    ) -> None:
        super().__init__()

        # self.subject_block = SubjectBlock(in_channels,in_channels,n_subject)
        self.subject_block = OriginalConvBlock(in_dim=in_channels,out_dim=in_channels)
        self.blocks = nn.Sequential(
            OriginalConvBlock(in_dim=in_channels, out_dim=hid_dim),
            OriginalConvBlock(in_dim=hid_dim, out_dim=hid_dim),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # X = self.subject_block(X,subject_idxs)
        X = self.subject_block(X)
        X = self.blocks(X)

        return self.head(X)

class ConvClassifier3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 256,
        hid_dim2: int = 128,
        n_subject: int = 4,
    ) -> None:
        super().__init__()

        # self.subject_block = SubjectBlock(in_channels,in_channels,n_subject)
        self.subject_block = OriginalConvBlock(in_dim=in_channels,out_dim=in_channels)
        self.blocks = nn.Sequential(
            OriginalConvBlock(in_dim=in_channels, out_dim=hid_dim),
            OriginalConvBlock(in_dim=hid_dim, out_dim=hid_dim2),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim2, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # X = self.subject_block(X,subject_idxs)
        X = self.subject_block(X)
        X = self.blocks(X)

        return self.head(X)

class OriginalConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        nn.init.kaiming_uniform_(self.conv0.weight)
        nn.init.zeros_(self.conv0.bias)    # バイアスの初期値を設定
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv0.bias)    # バイアスの初期値を設定
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)