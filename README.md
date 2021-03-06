# TSAI-END2.0-Assignment-11

1. Follow the similar strategy as we did in our baby-steps-code (Links to an external site.), but replace GRU with LSTM. In your code you must:
    * Perform 1 full feed forward step for the encoder manually
    * Perform 1 full feed forward step for the decoder manually.
    * You can use any of the 3 attention mechanisms that we discussed. 

2. Explain your steps in the readme file and
3. Submit the assignment asking for these things:
    * Link to the readme file that must explain Encoder/Decoder Feed-forward manual steps and the attention mechanism that you have used - 500 pts
    * Copy-paste (don't redirect to github), the Encoder Feed Forward steps for 2 words - 250 pts
    * Copy-paste (don't redirect to github), the Decoder Feed Forward steps for 2 words - 250 pts

## Solution

![image](https://user-images.githubusercontent.com/46154140/126875664-766f1067-792f-468e-9934-c14ba5277986.png)


We won't be covering the data preperation steps this time as the main focus is not understand the how we actually come up with encoder, decoder classes with attention by taking baby steps.

## The Encoder Model Steps - 

![image](https://user-images.githubusercontent.com/46154140/126875601-4aeefacf-97e8-4cf5-842b-b17607b740d6.png)


We will build our encoder with lstm. 

But before building any of embedding layers or lstm layer we need to convert our text inputs and outputs into some sort of numeric values to be processed by the embedding and lstm layers.

Currently our sample is list of french and english sentences.

```
Sample
['tu es vraiment embetante .', 'you re really annoying .']
```


### we can convert our sentences into list of indices and these indices can be further converted into tensors as the input to the pytorch's embedding layer is tensors
```python
input_sentence = sample[0]          # our input sentence
target_sentence = sample[1]         # output sentence

input_indices = [input_lang.word2index[word] for word in input_sentence.split(' ')]   #convert the texts into indices
target_indices = [output_lang.word2index[word] for word in target_sentence.split(' ')]   #convert the texts into indices

print(f'Input sentence indices {input_indices}, Output sentence indices {target_indices}')
```

```
Input sentence indices [210, 211, 957, 2800, 5], Output sentence indices [129, 78, 522, 344, 4]
```

### We also need to add the EOS token i.e END OF SENTENCE token to these list of indices so let's append that in our input and target indices

```python
input_indices.append(EOS_token)
target_indices.append(EOS_token)
print(f'Input sentence indices after adding EOS {input_indices}, Output sentence indices after adding EOS {target_indices}')
```

### but these are still indices so we need to convert them into tensors and move to cuda

```python
input_tensor = torch.tensor(input_indices, dtype=torch.long, device= device)
output_tensor = torch.tensor(target_indices, dtype=torch.long, device= device)
```


### we need to  also send our layers to cuda now as we are sending our tensors to cuda

### so now finally we can create our embedding layer, the input to the embedding layer is going to be the unique words or indices in our dataset and the embedding dimension or the hidden dimensions for the embedding layer.

```python
input_size = input_lang.n_words     # unique words in our  dataset
hidden_size = 256                   # we are keeping it as 256
embedding = nn.Embedding(input_size, hidden_size).to(device)
lstm = nn.LSTM(hidden_size, hidden_size,batch_first=True).to(device)
```

```python
hidden = torch.zeros(1, 1, 256, device=device).float()
cell_state = torch.zeros(1, 1, 256, device=device).float()

hidden = (hidden, cell_state)   # our hidden state will be hidden and cell_state in lstm
```
### We are working with 1 sample, but we would be working for a batch. We have to  fix that by converting our input_tensor into a fake batch

```python
embedded_input = embedding(input_tensor[0].view(-1, 1))
output, hidden = lstm(embedded_input,hidden)
print(output)
```
### encoder output for first word
```
tensor([[[ 2.4076e-01, -1.5498e-01, -1.6234e-03,  5.2397e-02,  1.9462e-04,
          -7.8092e-02, -5.6412e-02,  1.5028e-01, -1.1837e-01, -1.8692e-01,
           1.5497e-01, -1.2667e-01, -8.5491e-02,  1.6603e-01,  1.1597e-01,
          -1.5150e-01, -1.0964e-02,  1.2287e-02,  6.1560e-02, -1.9707e-01,
          -1.1195e-01,  1.7088e-01,  9.3712e-02,  1.8934e-01,  1.9608e-01,
           1.3662e-01, -1.6369e-01,  3.1864e-02,  2.0154e-01,  5.1636e-02,
           1.0000e-01,  1.4359e-01,  8.5468e-02,  6.4508e-02,  1.5932e-01,
          -1.2965e-01,  4.7318e-02, -7.3963e-02,  1.0462e-02,  3.9994e-02,
          -6.7153e-02, -1.0323e-01,  3.7370e-02,  1.0886e-01,  1.2317e-01,
           9.2645e-02,  2.8733e-01, -7.3676e-02, -1.3076e-01,  8.8094e-02,
           2.7493e-02,  1.9797e-01, -8.2009e-02, -1.1568e-01, -8.8958e-02,
          -1.2866e-02, -7.5198e-03,  1.1012e-01, -8.4974e-02,  1.1324e-01,
          -2.1695e-02, -9.6906e-02,  9.0477e-02, -1.9291e-01, -3.7823e-02,
           4.5775e-02,  1.4635e-01, -3.4157e-02,  2.6989e-02,  1.4280e-02,
          -1.0981e-01, -2.1248e-02,  2.8733e-01,  1.3537e-01,  7.3512e-02,
          -1.6476e-01,  2.7395e-02, -8.7181e-02, -3.3847e-02, -2.0818e-01,
           9.4704e-02,  4.6051e-02, -2.3448e-01, -3.8870e-02, -8.9110e-02,
           7.4304e-02, -2.2399e-02,  1.5103e-01, -4.3061e-02,  1.9708e-01,
           7.3660e-02, -2.3983e-01,  2.0101e-02, -8.2841e-02,  1.5380e-01,
          -5.1349e-02, -6.2672e-02,  9.1610e-02,  2.3617e-01, -1.4249e-01,
          -8.5370e-02,  3.4222e-02,  3.4324e-01,  4.8317e-02,  2.0338e-01,
          -1.4371e-02,  2.7858e-01,  1.4037e-01, -2.1064e-01, -4.7812e-02,
          -1.5648e-01,  1.2263e-01, -1.3773e-01,  2.3947e-02,  3.0776e-02,
          -1.5503e-01,  4.3023e-02,  6.0326e-03, -1.5282e-01, -4.8718e-02,
           3.0976e-02, -8.8542e-02, -1.1679e-01,  7.8672e-02,  6.4512e-02,
          -3.5573e-02, -5.0072e-02,  1.6683e-01,  5.5284e-02,  7.2355e-02,
           1.7642e-01, -3.1898e-02, -8.9423e-02,  6.7456e-02, -1.6365e-01,
          -2.1344e-01,  4.3441e-02, -1.1323e-01, -9.1896e-02, -1.4287e-01,
           1.2336e-01, -3.7688e-02, -3.3104e-02,  2.5446e-02, -3.1569e-02,
          -2.0750e-01, -2.3146e-01,  3.3670e-02, -1.2407e-01,  1.2140e-03,
          -5.1325e-02,  2.0116e-01, -5.8200e-02,  2.0444e-02, -6.7147e-02,
           3.7694e-02,  6.7206e-02,  3.4124e-02,  9.8360e-02,  1.0678e-01,
           1.8379e-01, -1.5139e-01,  6.2355e-02, -7.7538e-02,  5.5460e-02,
           7.0089e-02,  1.0031e-01,  3.0221e-01,  1.3953e-01, -2.8243e-01,
          -1.3520e-01, -4.2707e-03, -2.6139e-02, -6.4172e-02,  7.6386e-02,
          -3.1015e-02,  1.5951e-01, -1.0437e-01, -1.8963e-01, -8.9519e-02,
          -6.9490e-03,  1.3250e-01,  3.2258e-01, -1.0965e-01,  2.4924e-01,
          -9.9221e-02,  1.0442e-01,  3.1110e-02,  1.8240e-01, -7.6132e-02,
          -3.9342e-03, -1.0213e-01, -3.6908e-02, -1.3271e-01,  1.3106e-01,
          -7.8268e-03,  4.2335e-02,  1.6364e-01,  1.2239e-01,  1.5024e-01,
           6.6392e-02,  1.1167e-01, -2.2226e-02, -1.8695e-01,  1.0681e-01,
           7.7427e-02, -6.5347e-02, -1.0100e-01, -7.4946e-02, -2.3401e-01,
          -3.9985e-02,  4.3384e-02, -2.7060e-01,  2.2764e-02, -2.1711e-02,
          -5.8880e-02,  1.5366e-02, -5.7818e-02, -1.2035e-01, -3.7164e-02,
           6.2870e-02, -1.4142e-01, -6.6594e-02, -1.6556e-01,  3.8144e-02,
          -6.0452e-02,  1.9686e-02,  1.0039e-03, -6.9088e-02, -4.3797e-02,
           1.2419e-01, -4.4372e-02, -2.8868e-02,  1.8041e-01,  1.2856e-01,
           1.0490e-01, -9.1389e-02,  6.2437e-02,  1.4530e-01,  3.9922e-02,
           3.1193e-04, -1.2830e-01,  7.0849e-03, -2.5593e-02, -1.1787e-01,
           1.2390e-01, -3.3716e-02,  2.4121e-02,  8.7229e-02,  1.0765e-01,
           7.3454e-02,  1.9979e-02, -4.3594e-02,  3.3040e-02,  4.2820e-02,
          -2.0540e-01]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)
   ```
   
   ### encoder output for 2nd word
   
 ```python
embedded_input = embedding(input_tensor[1].view(-1, 1))
output, hidden = lstm(embedded_input,hidden)
print(output)
```
 ```
  tensor([[[ 0.2339, -0.0017, -0.0042,  0.0524, -0.0964, -0.0559, -0.0599,
           0.0624, -0.0714, -0.0786,  0.3199, -0.1077,  0.0973,  0.0114,
          -0.1098,  0.1671, -0.0600,  0.0229, -0.0159,  0.0076, -0.1928,
           0.0160,  0.2361,  0.1987,  0.0062,  0.0202, -0.0977,  0.0616,
           0.1214, -0.0750, -0.1216,  0.1066,  0.1933, -0.0035, -0.0195,
          -0.0339,  0.0037, -0.1564,  0.1146,  0.0745, -0.0567,  0.0763,
           0.1159, -0.0094, -0.1596,  0.1760,  0.1389, -0.1676, -0.1478,
          -0.0837,  0.1357,  0.0158, -0.1164,  0.1199,  0.0379, -0.1373,
           0.1390, -0.1488, -0.1375,  0.0290,  0.1032,  0.1420,  0.0140,
          -0.0190,  0.2117, -0.1028, -0.0743,  0.0445, -0.2696, -0.0879,
          -0.1944, -0.0653,  0.0020,  0.2300,  0.1352, -0.1839,  0.1149,
          -0.0487, -0.0482, -0.2032,  0.1133,  0.1052,  0.0039, -0.0917,
          -0.1415,  0.1489, -0.0423, -0.0036, -0.1235,  0.1452, -0.1122,
           0.0377, -0.0550, -0.3567, -0.0980, -0.2446, -0.1848, -0.0810,
          -0.0575, -0.0192, -0.0163,  0.0211,  0.1755,  0.1736,  0.1075,
          -0.0267,  0.1033,  0.0317, -0.3045, -0.1515, -0.0450, -0.0794,
          -0.1771,  0.1330, -0.1426,  0.0813,  0.0832,  0.1028, -0.0111,
          -0.0551,  0.0226,  0.0635, -0.0501,  0.1501,  0.0105, -0.0828,
          -0.0981, -0.2145,  0.0057,  0.0015,  0.2153,  0.0346, -0.2975,
           0.0933, -0.1655,  0.0111,  0.0260, -0.2760,  0.0264, -0.1104,
          -0.1770, -0.2532, -0.0420,  0.1496,  0.1998,  0.1305,  0.1530,
          -0.0017, -0.0900,  0.0328,  0.0773,  0.0836, -0.0258,  0.1808,
           0.1210, -0.0303, -0.0902, -0.0934,  0.0547, -0.0950, -0.0177,
          -0.0079, -0.0784,  0.1236,  0.0340, -0.1753,  0.1814,  0.0170,
           0.0354, -0.2281,  0.0263, -0.1244,  0.1089, -0.0363, -0.1026,
          -0.0654, -0.0943, -0.0956, -0.0737,  0.1282, -0.0744,  0.2038,
           0.2450, -0.1364,  0.0141,  0.0276,  0.3046,  0.1345,  0.0709,
          -0.0331, -0.0047, -0.1029,  0.1493, -0.1263, -0.0638,  0.1956,
           0.0785,  0.1059, -0.0267,  0.0100,  0.1528,  0.0669,  0.0959,
          -0.1991,  0.0874,  0.4115, -0.2484, -0.0045,  0.0049,  0.0379,
           0.0044,  0.0955, -0.0571,  0.0864,  0.1966, -0.2132,  0.0809,
           0.1142,  0.0559, -0.1097, -0.0691,  0.0270, -0.0623, -0.1139,
          -0.0381,  0.2527,  0.2476, -0.3246,  0.0206,  0.1161,  0.1503,
          -0.0854,  0.1850,  0.2034, -0.2354, -0.0958, -0.1284, -0.0255,
          -0.1540,  0.2379, -0.2576,  0.0650,  0.0924,  0.0234, -0.1352,
           0.0990,  0.0326,  0.0108, -0.0959,  0.2308,  0.1134, -0.0223,
           0.1880, -0.2077,  0.1359, -0.2973]]], device='cuda:0',
       grad_fn=<CudnnRnnBackward>)
   ```


## Decoder With Attention Steps

![image](https://user-images.githubusercontent.com/46154140/126875628-da71b010-aeb8-413c-838e-c499474340b7.png)


* First input to the decoder will be SOS_token, later inputs would be the words it predicted (unless we implement teacher forcing)
* decoder/LSTM's hidden state will be initialized with the encoder's last hidden state 
* we will use lstm's hidden state and last prediction to generate attention weight using a FC layer. 
* this attention weight will be used to weigh the encoder_outputs using batch matric multiplication. This will give us a NEW view on how to look at encoder_states.
* this attention applied encoder_states will then be concatenated with the input, and then sent a linear layer and _then_ sent to the LSTM. 
* LSTM's output will be sent to a FC layer to predict one of the output_language words

![image](https://user-images.githubusercontent.com/46154140/126875536-3494a954-a810-456d-9058-c71f897f729b.png)

```python
decoder_input = torch.tensor([[SOS_token]], device=device)
decoder_hidden = encoder_hidden
print(encoder_hidden[0])
output_size = output_lang.n_words
embedding = nn.Embedding(output_size, 256).to(device)
embedded = embedding(decoder_input)
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0][0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
input_to_gru_layer = nn.Linear(256 * 2, 256).to(device)
input_to_gru = input_to_gru_layer(torch.cat((embedded[0], attn_applied[0]), 1))
gru = nn.LSTM(256, 256,batch_first=True).to(device)
input_to_gru = input_to_gru.unsqueeze(0)
output, decoder_hidden = lstm(input_to_gru)
output_word_layer = nn.Linear(256, output_lang.n_words).to(device)
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
top_value, top_index = output.data.topk(1)
output_lang.index2word[top_index.item()]
```
### Output for step-1 of decoder


```
tensor([[[-2.1945e-01, -2.3019e-02,  9.8185e-02,  1.6685e-01, -1.2654e-02,
          -2.7083e-02,  4.2433e-02,  1.8986e-01, -1.0103e-01,  7.3739e-02,
          -3.0946e-01,  2.1215e-01, -1.5305e-01, -6.0538e-02, -2.3007e-02,
          -7.3478e-02, -3.7363e-01,  1.5101e-01,  1.4577e-01, -3.5632e-01,
           6.1078e-02, -5.8246e-02,  7.6801e-03,  1.2799e-01,  4.8085e-04,
           1.4507e-01, -9.8171e-02, -3.2640e-01,  1.9340e-01,  2.8965e-02,
           6.1186e-02,  1.6541e-01, -2.6627e-02, -6.6439e-02,  6.3617e-02,
           3.2684e-01, -5.5324e-02,  1.7559e-01,  1.7215e-01,  6.6446e-03,
          -3.9292e-02,  2.5475e-01, -2.1970e-01, -1.6074e-01, -1.5894e-01,
           9.7396e-02, -2.5441e-01,  1.3783e-01,  9.1362e-02, -3.5806e-02,
           1.0899e-02,  1.1908e-02, -9.2347e-02,  1.9092e-02, -1.3629e-01,
          -1.6168e-01, -3.2410e-03, -8.4668e-02,  1.4880e-01,  1.0728e-01,
          -1.5351e-01,  1.5443e-01,  1.1364e-01,  4.5125e-02, -3.8314e-02,
          -7.1266e-02, -2.2348e-01,  9.8422e-02, -1.5066e-02, -1.3242e-01,
          -1.2266e-01,  6.0875e-02, -2.2348e-01, -3.4858e-02, -5.5297e-02,
          -4.9570e-02,  1.4237e-01, -3.6150e-02,  8.5620e-02,  1.2860e-02,
           1.5205e-01,  1.0252e-01,  1.0062e-01, -2.9758e-02,  7.8722e-02,
          -6.3877e-02,  1.1527e-01,  9.5793e-02, -1.2630e-01, -3.7370e-02,
          -1.1360e-01,  5.6713e-02,  6.5998e-02,  8.1269e-02, -1.6767e-01,
           1.4050e-02, -6.1534e-02,  1.8212e-01,  1.1625e-01,  1.1150e-01,
           1.0402e-01, -3.3271e-01, -7.9771e-02,  5.4884e-02, -9.3756e-02,
          -2.9038e-02,  8.4211e-02,  2.0005e-02,  1.4679e-01,  7.7400e-03,
           6.2246e-02,  9.9347e-02, -1.0928e-01,  1.9505e-01,  1.2635e-01,
           4.8332e-02, -5.1404e-02,  8.9629e-02,  5.5809e-02,  2.5256e-01,
          -2.1589e-02, -1.5422e-01,  1.1892e-01, -5.1064e-02,  1.3224e-01,
           2.6907e-02,  4.1407e-02,  1.4086e-01,  6.6508e-02, -1.9089e-01,
           6.5447e-02,  1.0057e-01, -1.5916e-01, -1.3148e-01,  2.3614e-02,
           1.9406e-01,  1.4514e-01,  1.0616e-01,  3.1922e-01, -6.1377e-02,
           9.1087e-02, -2.9377e-01,  1.9261e-01,  8.3294e-02,  2.4683e-01,
           1.8225e-01,  9.5389e-02, -1.2005e-02,  8.7374e-03,  4.7688e-02,
          -3.7286e-02,  1.4689e-01, -1.0643e-01, -5.6674e-02, -1.3387e-01,
           8.1543e-02,  8.2250e-02, -1.7860e-01,  4.5327e-02, -5.4008e-02,
          -1.4741e-01, -1.6425e-01, -5.9389e-02,  1.2713e-01, -1.1900e-01,
           4.7389e-03,  4.5305e-02, -1.4798e-04,  1.2189e-01,  1.7055e-01,
          -1.5140e-01, -2.7196e-02,  9.0977e-02, -4.3800e-02, -2.3392e-01,
          -2.3702e-02, -1.9017e-01, -9.4101e-02, -4.5906e-02,  1.2925e-01,
           2.2964e-01,  1.7486e-01, -4.1400e-02, -1.1734e-01, -5.8956e-02,
          -9.2097e-02, -2.3074e-01, -6.5407e-02, -6.4783e-02,  1.3415e-01,
          -1.9481e-01, -6.5646e-02, -1.9758e-01, -4.6074e-02,  9.5178e-03,
           1.4566e-01,  2.1166e-02, -7.2544e-03, -1.9795e-01, -1.3912e-02,
           6.1571e-03,  1.7102e-01,  2.6034e-02, -9.4772e-02, -1.1633e-01,
           1.7858e-01, -4.4807e-02, -6.9204e-02,  3.0569e-01, -5.3852e-02,
          -1.7511e-02,  1.5893e-01,  1.2710e-01, -1.0841e-01, -1.2423e-01,
          -1.7166e-01, -1.2376e-01, -7.9423e-03,  3.0128e-01, -2.0261e-01,
           1.1896e-01, -2.9068e-03,  9.4861e-03,  1.7456e-01, -7.0413e-02,
          -5.4781e-02,  6.0414e-03,  1.5817e-01,  2.5081e-01,  2.3560e-02,
           9.4412e-02,  1.1055e-01,  3.8223e-02,  5.3875e-02, -2.2350e-01,
          -1.1386e-01, -2.3864e-01, -2.9456e-01, -1.1399e-01, -2.9223e-01,
           2.5128e-01,  3.3531e-02, -1.6719e-01, -4.8463e-02, -3.6843e-02,
           7.0752e-02, -2.2991e-01, -5.7096e-02,  2.9330e-02, -9.5623e-03,
          -1.0744e-02,  4.0849e-04, -9.1280e-02,  6.8897e-02, -3.2978e-02,
           1.6944e-01]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)
psyched
```

### output for step-2
```python
embedding = nn.Embedding(output_size, 256).to(device)
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_gru_layer = nn.Linear(256 * 2, 256).to(device)
lstm =  nn.LSTM(256, 256,batch_first=True).to(device)
output_word_layer = nn.Linear(256, output_lang.n_words).to(device)


decoder_input = torch.tensor([[SOS_token]], device=device)
decoder_hidden = encoder_hidden
print(encoder_hidden[0])
output_size = output_lang.n_words
embedded = embedding(decoder_input)
attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0][0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
input_to_gru = input_to_gru_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_gru = input_to_gru.unsqueeze(0)
output, decoder_hidden = lstm(input_to_gru)
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
top_value, top_index = output.data.topk(1)
output_lang.index2word[top_index.item()], attn_weights
```
```
tensor([[[-2.1945e-01, -2.3019e-02,  9.8185e-02,  1.6685e-01, -1.2654e-02,
          -2.7083e-02,  4.2433e-02,  1.8986e-01, -1.0103e-01,  7.3739e-02,
          -3.0946e-01,  2.1215e-01, -1.5305e-01, -6.0538e-02, -2.3007e-02,
          -7.3478e-02, -3.7363e-01,  1.5101e-01,  1.4577e-01, -3.5632e-01,
           6.1078e-02, -5.8246e-02,  7.6801e-03,  1.2799e-01,  4.8085e-04,
           1.4507e-01, -9.8171e-02, -3.2640e-01,  1.9340e-01,  2.8965e-02,
           6.1186e-02,  1.6541e-01, -2.6627e-02, -6.6439e-02,  6.3617e-02,
           3.2684e-01, -5.5324e-02,  1.7559e-01,  1.7215e-01,  6.6446e-03,
          -3.9292e-02,  2.5475e-01, -2.1970e-01, -1.6074e-01, -1.5894e-01,
           9.7396e-02, -2.5441e-01,  1.3783e-01,  9.1362e-02, -3.5806e-02,
           1.0899e-02,  1.1908e-02, -9.2347e-02,  1.9092e-02, -1.3629e-01,
          -1.6168e-01, -3.2410e-03, -8.4668e-02,  1.4880e-01,  1.0728e-01,
          -1.5351e-01,  1.5443e-01,  1.1364e-01,  4.5125e-02, -3.8314e-02,
          -7.1266e-02, -2.2348e-01,  9.8422e-02, -1.5066e-02, -1.3242e-01,
          -1.2266e-01,  6.0875e-02, -2.2348e-01, -3.4858e-02, -5.5297e-02,
          -4.9570e-02,  1.4237e-01, -3.6150e-02,  8.5620e-02,  1.2860e-02,
           1.5205e-01,  1.0252e-01,  1.0062e-01, -2.9758e-02,  7.8722e-02,
          -6.3877e-02,  1.1527e-01,  9.5793e-02, -1.2630e-01, -3.7370e-02,
          -1.1360e-01,  5.6713e-02,  6.5998e-02,  8.1269e-02, -1.6767e-01,
           1.4050e-02, -6.1534e-02,  1.8212e-01,  1.1625e-01,  1.1150e-01,
           1.0402e-01, -3.3271e-01, -7.9771e-02,  5.4884e-02, -9.3756e-02,
          -2.9038e-02,  8.4211e-02,  2.0005e-02,  1.4679e-01,  7.7400e-03,
           6.2246e-02,  9.9347e-02, -1.0928e-01,  1.9505e-01,  1.2635e-01,
           4.8332e-02, -5.1404e-02,  8.9629e-02,  5.5809e-02,  2.5256e-01,
          -2.1589e-02, -1.5422e-01,  1.1892e-01, -5.1064e-02,  1.3224e-01,
           2.6907e-02,  4.1407e-02,  1.4086e-01,  6.6508e-02, -1.9089e-01,
           6.5447e-02,  1.0057e-01, -1.5916e-01, -1.3148e-01,  2.3614e-02,
           1.9406e-01,  1.4514e-01,  1.0616e-01,  3.1922e-01, -6.1377e-02,
           9.1087e-02, -2.9377e-01,  1.9261e-01,  8.3294e-02,  2.4683e-01,
           1.8225e-01,  9.5389e-02, -1.2005e-02,  8.7374e-03,  4.7688e-02,
          -3.7286e-02,  1.4689e-01, -1.0643e-01, -5.6674e-02, -1.3387e-01,
           8.1543e-02,  8.2250e-02, -1.7860e-01,  4.5327e-02, -5.4008e-02,
          -1.4741e-01, -1.6425e-01, -5.9389e-02,  1.2713e-01, -1.1900e-01,
           4.7389e-03,  4.5305e-02, -1.4798e-04,  1.2189e-01,  1.7055e-01,
          -1.5140e-01, -2.7196e-02,  9.0977e-02, -4.3800e-02, -2.3392e-01,
          -2.3702e-02, -1.9017e-01, -9.4101e-02, -4.5906e-02,  1.2925e-01,
           2.2964e-01,  1.7486e-01, -4.1400e-02, -1.1734e-01, -5.8956e-02,
          -9.2097e-02, -2.3074e-01, -6.5407e-02, -6.4783e-02,  1.3415e-01,
          -1.9481e-01, -6.5646e-02, -1.9758e-01, -4.6074e-02,  9.5178e-03,
           1.4566e-01,  2.1166e-02, -7.2544e-03, -1.9795e-01, -1.3912e-02,
           6.1571e-03,  1.7102e-01,  2.6034e-02, -9.4772e-02, -1.1633e-01,
           1.7858e-01, -4.4807e-02, -6.9204e-02,  3.0569e-01, -5.3852e-02,
          -1.7511e-02,  1.5893e-01,  1.2710e-01, -1.0841e-01, -1.2423e-01,
          -1.7166e-01, -1.2376e-01, -7.9423e-03,  3.0128e-01, -2.0261e-01,
           1.1896e-01, -2.9068e-03,  9.4861e-03,  1.7456e-01, -7.0413e-02,
          -5.4781e-02,  6.0414e-03,  1.5817e-01,  2.5081e-01,  2.3560e-02,
           9.4412e-02,  1.1055e-01,  3.8223e-02,  5.3875e-02, -2.2350e-01,
          -1.1386e-01, -2.3864e-01, -2.9456e-01, -1.1399e-01, -2.9223e-01,
           2.5128e-01,  3.3531e-02, -1.6719e-01, -4.8463e-02, -3.6843e-02,
           7.0752e-02, -2.2991e-01, -5.7096e-02,  2.9330e-02, -9.5623e-03,
          -1.0744e-02,  4.0849e-04, -9.1280e-02,  6.8897e-02, -3.2978e-02,
           1.6944e-01]]], device='cuda:0', grad_fn=<CudnnRnnBackward>)
('outraged',
 tensor([[0.0309, 0.0497, 0.0853, 0.1070, 0.0793, 0.1405, 0.0956, 0.1807, 0.1516,
          0.0794]], device='cuda:0', grad_fn=<SoftmaxBackward>))
   ```
