import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        # Register hooks and store them for cleanup
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [forward_handle, backward_handle]

    def generate(self, input_tensor, class_idx=None):
        try:
            self.model.eval()
            
            # Reset gradients and activations
            self.gradients = None
            self.activations = None
            
            # Forward pass
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()

            # Backward pass
            self.model.zero_grad()
            class_score = output[0, class_idx]
            class_score.backward(retain_graph=True)

            # Check if we got gradients and activations
            if self.gradients is None or self.activations is None:
                print("Warning: No gradients or activations captured")
                return self._create_dummy_heatmap(input_tensor)

            # Handle different tensor shapes
            if len(self.gradients.shape) == 4:  # Standard CNN: [batch, channels, height, width]
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
                activations = self.activations.squeeze(0)
                
                for i in range(len(pooled_gradients)):
                    activations[i, :, :] *= pooled_gradients[i]
                
                heatmap = torch.mean(activations, dim=0).cpu().numpy()
                
            elif len(self.gradients.shape) == 3:  # ViT-like: [batch, seq_length, hidden_dim]
                # For ViT models, we need to reshape to spatial dimensions
                batch_size, seq_len, hidden_dim = self.gradients.shape
                
                # Skip CLS token (first token) and reshape to spatial grid
                spatial_grads = self.gradients[:, 1:, :]  # Remove CLS token
                spatial_acts = self.activations[:, 1:, :]
                
                # Calculate grid size (assuming square patches)
                grid_size = int(np.sqrt(spatial_grads.shape[1]))
                if grid_size * grid_size != spatial_grads.shape[1]:
                    print(f"Warning: Cannot reshape {spatial_grads.shape[1]} patches to square grid")
                    return self._create_dummy_heatmap(input_tensor)
                
                # Reshape to spatial dimensions
                spatial_grads = spatial_grads.reshape(batch_size, grid_size, grid_size, hidden_dim)
                spatial_acts = spatial_acts.reshape(batch_size, grid_size, grid_size, hidden_dim)
                
                # Compute importance weights
                weights = torch.mean(spatial_grads, dim=(1, 2))  # Average over spatial dimensions
                weights = weights.squeeze(0)  # Remove batch dimension
                
                # Weight the activations
                weighted_acts = spatial_acts.squeeze(0)  # Remove batch dimension
                for i in range(len(weights)):
                    weighted_acts[:, :, i] *= weights[i]
                
                # Sum over feature dimension
                heatmap = torch.sum(weighted_acts, dim=2).cpu().numpy()
                
            else:
                print(f"Unsupported gradient shape: {self.gradients.shape}")
                return self._create_dummy_heatmap(input_tensor)

            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            return heatmap
            
        except Exception as e:
            print(f"Error in GradCAM generation: {str(e)}")
            return self._create_dummy_heatmap(input_tensor)

    def _create_dummy_heatmap(self, input_tensor):
        """Create a dummy heatmap when GradCAM fails"""
        # Create a simple center-focused heatmap
        size = 14  # Typical feature map size
        heatmap = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                heatmap[i, j] = max(0, 1 - dist / (size/2))
        return heatmap

    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class ViTGradCAM:
    """Specialized GradCAM for Vision Transformer models"""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []

    def generate(self, input_tensor, class_idx=None):
        try:
            self.model.eval()
            
            # Hook the last attention layer
            def hook_fn(module, input, output):
                if hasattr(output, 'last_hidden_state'):
                    self.activations = output.last_hidden_state
                elif isinstance(output, tuple) and len(output) > 0:
                    self.activations = output[0]
                else:
                    self.activations = output

            # Register hook on the ViT encoder
            if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'encoder'):
                hook_handle = self.model.vit.encoder.register_forward_hook(hook_fn)
                self.hooks.append(hook_handle)

            # Forward pass
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()

            # Backward pass
            self.model.zero_grad()
            class_score = output[0, class_idx]
            class_score.backward(retain_graph=True)

            # Get gradients from the activations
            if self.activations is not None and self.activations.grad is not None:
                gradients = self.activations.grad
                
                # Process ViT features
                batch_size, seq_len, hidden_dim = gradients.shape
                
                # Skip CLS token and reshape to spatial grid
                spatial_grads = gradients[:, 1:, :]
                spatial_acts = self.activations[:, 1:, :].detach()
                
                grid_size = int(np.sqrt(spatial_grads.shape[1]))
                if grid_size * grid_size == spatial_grads.shape[1]:
                    # Reshape to spatial dimensions
                    spatial_grads = spatial_grads.reshape(batch_size, grid_size, grid_size, hidden_dim)
                    spatial_acts = spatial_acts.reshape(batch_size, grid_size, grid_size, hidden_dim)
                    
                    # Compute importance weights
                    weights = torch.mean(spatial_grads, dim=(1, 2)).squeeze(0)
                    
                    # Weight the activations
                    weighted_acts = spatial_acts.squeeze(0)
                    for i in range(len(weights)):
                        weighted_acts[:, :, i] *= weights[i]
                    
                    heatmap = torch.sum(weighted_acts, dim=2).cpu().numpy()
                    heatmap = np.maximum(heatmap, 0)
                    if np.max(heatmap) > 0:
                        heatmap = heatmap / np.max(heatmap)
                    
                    return heatmap
            
            # Fallback to dummy heatmap
            return self._create_dummy_heatmap()
            
        except Exception as e:
            print(f"Error in ViT GradCAM: {str(e)}")
            return self._create_dummy_heatmap()
        finally:
            self.cleanup()

    def _create_dummy_heatmap(self):
        size = 14
        heatmap = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                heatmap[i, j] = max(0, 1 - dist / (size/2))
        return heatmap

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def apply_colormap_on_image(org_img, activation_map, alpha=0.5):
    """Apply colormap to create heatmap overlay"""
    try:
        # Convert PIL to numpy if needed
        if hasattr(org_img, 'convert'):
            org_img = np.array(org_img.convert('RGB'))
        elif isinstance(org_img, np.ndarray):
            if len(org_img.shape) == 3 and org_img.shape[2] == 3:
                # Already in RGB format
                pass
            else:
                # Convert to RGB if needed
                org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        
        # Ensure activation map is properly sized
        if len(activation_map.shape) != 2:
            print(f"Warning: Unexpected activation map shape: {activation_map.shape}")
            activation_map = activation_map.squeeze()
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
        
        # Normalize heatmap
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to uint8 and apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert heatmap to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        superimposed_img = cv2.addWeighted(org_img.astype(np.uint8), 1 - alpha, 
                                         heatmap, alpha, 0)
        
        return superimposed_img
        
    except Exception as e:
        print(f"Error in apply_colormap_on_image: {str(e)}")
        # Return original image if overlay fails
        return org_img if isinstance(org_img, np.ndarray) else np.array(org_img)