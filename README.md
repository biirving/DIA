# DIA
Decepticons in Alexandria - a repo for transformers. 



Install:
```
pip install DecepticonsInAlexandria
```


ViT example:
```
from DIAtransformers import vit

# arguments for vit (in order): 
# height
# width
# patch size
# dimension 
# number of output classes
# batch size

vit_model = vit(224, 224, 16, 512, 10, 1)
example_input = torch.randn(1, 3, 224, 224)

vit_mlp_output = vit_model(example_input)

print(vit_mlp_output)
```

timeSformer example:
```
from DIAtransformers import timeSformer

# arguments for timeSformer (in order): 
# height
# width
# number of frames
# patch size
# dimension 
# number of output classes
# batch size

tf_model = timeSformer(224, 224, 7, 16, 512, 10, 3)
video_input = torch.randn(3, 7, 3, 224, 224)

tf_mlp_output = tf_model(video_input)

print(tf_mlp_output)
```

So far, the finished implementations include the Vision Transformer
```
@misc{https://doi.org/10.48550/arxiv.2010.11929,
  doi = {10.48550/ARXIV.2010.11929},
  
  url = {https://arxiv.org/abs/2010.11929},
  
  author = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

And the TimeSformer models
```
@misc{https://doi.org/10.48550/arxiv.2102.05095,
  doi = {10.48550/ARXIV.2102.05095},
  
  url = {https://arxiv.org/abs/2102.05095},
  
  author = {Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Is Space-Time Attention All You Need for Video Understanding?},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
