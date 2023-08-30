# VPGNN: Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks
We provide the implementaion of VPGNN model, which is the source code for CIKM 2023 ADS paper
"Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks". 

The repository is organised as follows:
- data/: the directory of data sets, and it contains the Amazon data set as the example. 
- res/: the directory of saved models.
- data_process.py: data preperation for node features, lables, edges.
- pre_train.py: DGI pre-training
- prompt.py: make well initialized prompt be learnable vectors.
- final_model_gnn.py: integrate the prompt and gnn into the final model.
- main.py: prompt-based fine-tuning and final prediction.


## Requirements

  To install requirements:

    pip install -r requirements.txt


## Data
  In VPGNN directory, to unzip the datasets, run:
  
    unzip /data/Amazon.zip
    
  To generate node feature, label, edges, run:
  
    python data_process.py
    
    
## Train and test

  To pre-train the model in the paper:
  
    python pre_train.py
    
  Prompt-based fine-tuning and final prediction:
  
    python main.py
    


