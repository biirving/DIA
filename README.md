# DIA
Decepticons in Alexandria - a repo for some transformers


ViT example:
```
vit_model = vit(224, 224, 16, 512, 10, 1)
example_input = torch.randn(1, 3, 224, 224)

vit_mlp_output = vit_model(example_input)

print(vit_mlp_output)
```

timeSformer example:
```
tf_model = timeSformer(224, 224, 7, 16, 512, 10, 3)
input = torch.randn(3, 7, 3, 224, 224)

tf_mlp_output = tf_model(input)

print(tf_mlp_output)
```
