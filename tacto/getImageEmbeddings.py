import numpy as np
import torch
import gc 
from tacto.fcrn import fcrn

def getImageEmbeddings(model, images):
    # Adapted from original PointNetVLAD code

    if type(images) is not list:
        images = [images]
    
    embeddings_l = [None] * len(images)
    
    with torch.no_grad():
        for j, image in enumerate(images):
            embedding = model.image2embedding(image)
            embedding = embedding/np.linalg.norm(embedding, axis=1).reshape(-1, 1) # normalize embedding
            embeddings_l[j] = embedding

    embeddings_l = np.vstack(embeddings_l) # list to array (set_sz, output_dim)
    gc.collect()
    return embeddings_l



    def test(self, test_data):
        # test_data: tactile img 640 * 480
        # result: height map 640 * 480

        test_set = TestDataLoader(test_data)
        test_loader = torch.utils.data.DataLoader(test_set, **self.params)
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output, feature = self.model(data.float())
                # feature: 10 * 8 * 1024 -> (BS,81920)
                return feature
