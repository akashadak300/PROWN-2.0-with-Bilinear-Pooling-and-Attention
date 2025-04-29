import torch
import copy
import time

from torch.distributions import Beta
from torch.utils.data import DataLoader, Dataset
from aux import flatten_params, recover_flattened
from utils import evaluate_model
from tqdm import tqdm


def reject_data(data, f):
    return tuple(data[i][f] for i in range(len(data)))
        
def data_to_device(data, device='cpu'):
    return tuple(data[i].detach().to(device) for i in range(len(data)))
        
    
class BASE_DATASET(Dataset):
    def __init__(self, args):
        
        self.data = []
        
        self.N_base = 128 # Number of watermark images before rejection
        self.args = args
        
        if args.threshold is not None:
            size_ = 512
            test_dataset_for_eval, test_dataset_ = torch.utils.data.random_split(args.test_dataset, [size_, len(args.test_dataset) - size_])
            dataloader = DataLoader(test_dataset_for_eval, shuffle=False, batch_size=512)
            acc_model = evaluate_model(args.model, dataloader, args.device)
        
        ### Collect adversarial examples ###
        base_dataset = args.train_dataset if args.use_train else args.test_dataset
        adv_dataset = self.generate(base_dataset)
        
        ### Reject bad watermarks ###
        flatten_dict = flatten_params(args.model.parameters())
        init_params, init_indices = flatten_dict['params'], flatten_dict['indices']    

        model_copy = copy.deepcopy(args.model)
        model_copy.eval()
        model_copy.to(args.device)
        
        M = args.M // 2 if (args.sigma1 is not None and args.sigma2 is not None) else args.M
        for i in tqdm(range(M)):  
            if args.sigma1 is not None:
                delta = args.sigma1 * torch.randn_like(init_params)
                new_params = init_params + delta
                new_params_unfl = recover_flattened(new_params, init_indices, model_copy)

                for i, params in enumerate(model_copy.parameters()):
                    params.data = new_params_unfl[i].data
                
                if args.threshold is not None:
                    acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
                    
                    while abs(acc_model_copy - acc_model) > args.threshold:
                        delta = args.sigma1 * torch.randn_like(init_params)
                        new_params = init_params + delta
                        new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
                        
                        for i, params in enumerate(model_copy.parameters()):
                            params.data = new_params_unfl[i].data
                        
                        acc_model_copy = evaluate_model(model_copy, dataloader, args.device)
                        print("Accuracy gap:", abs(acc_model_copy - acc_model))
                
                update_adv_dataset = []
                for data in adv_dataset:
                    advs, labels = data[0], data[1]
                    advs, labels = advs.to(args.device), labels.to(args.device)
                    
                    with torch.no_grad():
                        logits = model_copy(advs)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    f = (predictions == labels).detach().cpu()
                    if f.float().sum():
                        update_adv_dataset.append(reject_data(data, f))
                        
                adv_dataset = update_adv_dataset
                
            
            if args.sigma2 is not None:
                delta = args.sigma2 * torch.randn_like(init_params)
                new_params = init_params + delta
                new_params_unfl = recover_flattened(new_params, init_indices, model_copy)
                for i, params in enumerate(model_copy.parameters()):
                    params.data = new_params_unfl[i].data
                    
                update_adv_dataset = []
                for data in adv_dataset:
                    advs, labels = data[0], data[1]
                    advs, labels = advs.to(args.device), labels.to(args.device)
                    
                    with torch.no_grad():
                        logits = model_copy(advs)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    f = (predictions != labels).detach().cpu()
                    if f.float().sum():
                        update_adv_dataset.append(reject_data(data, f))
                        
                adv_dataset = update_adv_dataset
                
        for data in adv_dataset:
            self.data.append(data_to_device(data))

        print("len dataset", len(self.data))
        print("reject coefficient", len(self.data) / self.N_base)
        
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def generate(self):
        raise NotImplementedError("Please Implement this method")
        


# Modified code    
class L_DATASET_(BASE_DATASET):
    def __init__(self, args, idxs=[]):
        self.idxs = idxs
        super().__init__(args)
    
    def generate(self, base_dataset):
        args = self.args
        I = 0
        adv_dataset = []
        
        while True:
            idx1 = torch.randint(0, len(base_dataset), (1,)).item()
            idx2 = torch.randint(0, len(base_dataset), (1,)).item()

            if idx1 == idx2:
                continue
            
            if (idx1 in self.idxs) and (idx2 in self.idxs):
                continue
            
            image1, label1 = base_dataset[idx1]
            image2, label2 = base_dataset[idx2]
            
            if label1 == label2:
                continue

            image1 = image1.repeat(args.batch, 1, 1, 1)
            image2 = image2.repeat(args.batch, 1, 1, 1)
            lambda_ = torch.rand(args.batch, 1, 1, 1)

            # --- NEW FIX ---
            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            lambda_ = lambda_.to(args.device)
            # --- END FIX ---

            
            # --- Modified for Bilinear Pooling below ---
            # Compute outer product channel-wise
            b, c, h, w = image1.shape
            image1_flat = image1.view(b, c, -1)  # (batch, channels, h*w)
            image2_flat = image2.view(b, c, -1)  # (batch, channels, h*w)
            
            bilinear = torch.bmm(image1_flat, image2_flat.transpose(1, 2))  # (batch, channels, channels)
            bilinear = bilinear.view(b, c * c, 1, 1)  # reshape to (batch, c*c, 1, 1)
            
            # Normalize bilinear output
            bilinear = torch.nn.functional.normalize(bilinear, dim=1)
            
            # bilinear = bilinear.to(args.device)
            bilinear = bilinear.to(args.device)  # <<<--- ADD THIS LINE

            # Simple projection to match original dimensions if needed
            proj = torch.nn.Conv2d(c * c, c, kernel_size=1).to(args.device)  # linear projection to reduce channels
            bilinear_projected = proj(bilinear)  # shape: (batch, c, 1, 1)
            bilinear_projected = torch.nn.functional.interpolate(bilinear_projected, size=(h, w), mode='bilinear')  # upsample to (h, w)
            
            images = (1 - lambda_) * image1 + lambda_ * image2 + 0.1 * bilinear_projected  # Add a small bilinear interaction
            # --- End of Bilinear Pooling modification ---

            images = images.to(args.device)
            lambda_ = lambda_.squeeze()
            
            with torch.no_grad():
                logits = args.model(images).detach()
                target_predictions = torch.argmax(logits, dim=-1)
            
            images, target_predictions = images.detach().cpu(), target_predictions.detach().cpu()
            f = (target_predictions != label1) * (target_predictions != label2)
            
            if f.float().sum():
                images, target_predictions = images[f], target_predictions[f]
                lambda_ = lambda_[f]

                adv_dataset.append((images, target_predictions, lambda_))
                I += 1
                self.idxs.append(idx1)
                self.idxs.append(idx2)
                
            if I == self.N_base:
                break
        
        return adv_dataset



class FINAL_DATASET(Dataset):
    def __init__(self, args):
        self.data = []
        
        times = []
        t = 0
        idxs = []
        while len(self.data) < args.N:
            start = time.time()
            dataset = L_DATASET_(args, idxs)
            idxs = dataset.idxs
            end = time.time()
            times.append(end - start)
            
            # TODO add another selection criterion
            for data in dataset:
                # Uncomment to use selection criterion via lambda_
                # lambda_ = data[2]
                # lambda_ = torch.abs(lambda_ - 0.5)
                # idx = torch.argmax(lambda_)
                
                idx = 0
                self.data.append(tuple(data[i][idx] for i in range(len(data))))
                               
                if len(self.data) == args.N:
                    break
            print("Elements in trigerset:", len(self.data))  
            t += 1
            
        print("ReSample times:", t)
        print("Average time:", sum(times) / len(times))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]