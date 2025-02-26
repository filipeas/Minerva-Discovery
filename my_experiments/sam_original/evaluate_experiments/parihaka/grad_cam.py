import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer, gpu_id:int = 0, multimask_output:bool=False):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.target_layer = target_layer
        self.multimask_output = multimask_output
        self.gradients = None
        self.activations = None

        # Register hooks
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hook the target layer
        target_layer = dict([*self.model.named_modules()])[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, batch):
        """ Make inference with model """
        return self.model(batch, multimask_output=self.multimask_output)
    
    def generate_cam(self, batch, label, backward_aproach:int = 1):
        # Ensure the model is in training mode
        self.model.eval()

        # Move batch and label to GPU
        batch['image'] = batch['image'].to(self.device)
        batch['label'] = batch['label'].to(self.device)
        batch['point_coords'] = batch['point_coords'].to(self.device)
        batch['point_labels'] = batch['point_labels'].to(self.device)
        label = label.to(self.device)

        # Enable the calculation of gradients explicitly
        with torch.enable_grad():
            # Forward pass
            output = self.forward([batch])

        # Backward pass
        self.model.zero_grad()

        if backward_aproach == 1:
            # 1) you need a global vision?
            output.backward(torch.ones_like(output), retain_graph=True) # or output.sum().backward(retain_graph=True) or output.mean().backward(retain_graph=True)
        elif backward_aproach == 2:
            # 2) Using the label (mask) to agregate the output (global focal vision)
            mask = label.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            localized_output = output * mask # keep only the values of the object region
            localized_score = localized_output.sum() # or .mean()
            localized_score.backward(retain_graph=True)
        elif backward_aproach == 3:
            # 3) Select a representative pixel (specific localized view), for exemple, the pixel with the highest activation in the annoted region
            mask = label.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            masked_output = output * mask
            # finding the pixel with hightest activation inside the anoted region
            # the function view(-1) "flatten" the tensor to 1D e .max(0) return the max value and the curresponding index
            max_value, idx = masked_output.view(-1).max(0)
            # Converte o índice (1D) de volta para coordenadas 2D (assumindo que output tem shape [1, 1, H, W])
            H, W = output.shape[2], output.shape[3]
            i = idx // W  # linha
            j = idx % W   # coluna
            # Propaga o gradiente para esse pixel específico
            output[0, 0, i, j].backward(retain_graph=True)

        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        height = batch['image'].shape[-2]
        width = batch['image'].shape[-1]
        cam = cv2.resize(cam, (width, height))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, output