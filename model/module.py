import lightning as L
import torch
from torch import nn
from torch.func import jvp
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.unet import MNISTUNet
from model.cvf import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from torchmetrics import MeanMetric


optimizer_map = {'adamW': AdamW, 'adam': Adam}


class MeanFlowModule(L.LightningModule):
    """
    Mean Flow training module following arxiv:2505.13447.

    Key difference from standard flow matching:
    - Model takes two timesteps (t, r) with r >= t
    - Loss target uses a JVP correction: u_tgt = v + (r - t) * du/dt
    - This trains the model to predict the *mean* velocity over [t, r],
      enabling better few-step (or one-step) generation at inference.
    - For r == t the JVP term vanishes and it reduces to standard FM.
    """
    def __init__(self, model_cfg: dict, train_cfg: dict):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = MNISTUNet(
            channels=model_cfg['channels'],
            num_residual_layers=model_cfg['num_residual_layers'],
            t_embed_dim=model_cfg['t_embed_dim'],
            y_embed_dim=model_cfg['y_embed_dim'],
        )

        self.eta = model_cfg['eta']
        # Fraction of batch where r=t (reduces to standard FM step, no JVP cost)
        self.flow_ratio = 0.5

        self.path = GaussianConditionalProbabilityPath(
            p_simple_shape=model_cfg['input_size'],
            alpha=LinearAlpha(),
            beta=LinearBeta(),
        )

        self.optimizer_type = optimizer_map[train_cfg['optimizer']]
        self.learning_rate = train_cfg['learning_rate']
        self.monitor = train_cfg['early_stopping_val']

        self.metrics = nn.ModuleDict()
        for split in ['train/', 'val/', 'test/']:
            self.metrics[split] = MeanMetric()

    def _adaptive_l2_loss(self, error: torch.Tensor, gamma: float = 0.5, c: float = 1e-3) -> torch.Tensor:
        """
        Adaptive L2 loss from the paper (eq. in Sec 3.2).
        Down-weights samples with large errors so training is more stable.
        """
        delta_sq = torch.mean(error ** 2, dim=(1, 2, 3))  # (bs,)
        p = 1.0 - gamma
        w = (delta_sq + c).pow(-p).detach()  # stop-gradient on weights
        return (w * delta_sq).mean()

    def model_step(self, batch, data_split, batch_idx):
        z, y = batch
        batch_size = z.shape[0]

        # CFG dropout: replace label with null token (10) with probability eta
        mask = torch.rand(batch_size) > self.eta
        y[mask] = 10

        # Sample two timesteps t <= r, both in [0, 1]
        # (t=current position on path, r=reference/target position closer to data)
        s1 = torch.rand(batch_size, 1, 1, 1, device=z.device)
        s2 = torch.rand(batch_size, 1, 1, 1, device=z.device)
        t = torch.minimum(s1, s2)
        r = torch.maximum(s1, s2)

        # For flow_ratio fraction of samples, collapse r=t (standard FM step)
        collapse = torch.rand(batch_size, device=z.device) < self.flow_ratio
        r = torch.where(collapse.view(-1, 1, 1, 1), t, r)

        # Interpolate: x_t = t*z + (1-t)*eps  (t=0 is noise, t=1 is data)
        eps = torch.randn_like(z)
        x_t = t * z + (1 - t) * eps
        v = z - eps  # instantaneous velocity (points toward data)

        # JVP: compute u = model(x_t, t, r) and dudt = d/dt[u] along the trajectory
        # Tangents: dx_t/dt = v (trajectory velocity), dt/dt = 1, dr/dt = 0
        # Result: dudt = ∂u/∂x_t * v + ∂u/∂t
        y_fixed = y  # y is not differentiated — captured in closure
        u, dudt = jvp(
            lambda x, t_, r_: self.model(x, t_, r_, y_fixed),
            (x_t, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True,  # needed so gradients flow back through u
        )

        # Mean flow target (stop-gradient): u_tgt = v + (r - t) * dudt
        # When r=t: reduces to v (standard FM)
        # When r>t: JVP correction makes target the *mean* velocity over [t,r]
        u_tgt = (v + (r - t) * dudt).detach()

        loss = self._adaptive_l2_loss(u - u_tgt)

        self.metrics[data_split](loss)
        self.log(f'{data_split}mf_loss', self.metrics[data_split], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, "train/", batch_idx)

    def validation_step(self, batch, batch_idx):
        self.model_step(batch, "val/", batch_idx)

    def test_step(self, batch, batch_idx):
        self.model_step(batch, "test/", batch_idx)

    def configure_optimizers(self):
        optimizer = self.optimizer_type(
            params=self.trainer.model.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )
        scheduler = ReduceLROnPlateau(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }


class FlowMatchingModule(L.LightningModule):
    def __init__(self, model_cfg: dict, train_cfg: dict):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        #load model
        self.model = MNISTUNet(
            channels = model_cfg['channels'],
            num_residual_layers = model_cfg['num_residual_layers'],
            t_embed_dim = model_cfg['t_embed_dim'],
            y_embed_dim = model_cfg['y_embed_dim'],
        )
        
        #probability to set label to null
        self.eta = model_cfg['eta']

        #initialize conditional probability path
        self.path = GaussianConditionalProbabilityPath(
            p_simple_shape = model_cfg['input_size'], #sampled Gaussian must match image size
            alpha = LinearAlpha(),
            beta = LinearBeta()
        )

        #set training parameters
        self.optimizer_type = optimizer_map[train_cfg['optimizer']]
        self.learning_rate = train_cfg['learning_rate']
        self.monitor = train_cfg['early_stopping_val']

        #initialize metrics as ModuleDict to enforce them as attributes of the LightningModule
        self.metrics = nn.ModuleDict()
        for split in ['train/', 'val/', 'test/']:
            self.metrics[split] = MeanMetric() #collect CFM loss

    def model_step(self, batch, data_split, batch_idx):

        # Step 1: Sample z,y from p_data
        z, y = batch
        batch_size = z.shape[0]

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) > self.eta
        y[mask] = 10

        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)

        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress and output loss
        u_t_theta = self.model(x, t, t, y)  # r=t for standard flow matching
        u_ref = self.path.conditional_vector_field(x, z, t)
        cfm_loss = torch.mean((u_t_theta - u_ref) ** 2)

        #log metrics
        self.metrics[data_split](cfm_loss)
        self.log(f'{data_split}cfm_loss', self.metrics[data_split], on_step=False, on_epoch=True, prog_bar=True)
        
        #loss required for backprop
        return cfm_loss


    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, "train/", batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model_step(batch, "val/", batch_idx)
    
    def test_step(self, batch, batch_idx):
        self.model_step(batch, "test/", batch_idx)

    def configure_optimizers(self):
        optimizer = self.optimizer_type(params=self.trainer.model.parameters(), 
                                        lr=self.learning_rate,
                                        amsgrad=True)
        
        #learning rate scheduler used in spliceai pytorch implementation
        scheduler = ReduceLROnPlateau(optimizer=optimizer)

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor
                }
            }
