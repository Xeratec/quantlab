# 
# simplecnn.py
# 
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# 
# Copyright (c) 2023 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, IBertForSequenceClassification


class iBert(nn.Module):
    def __init__(self, seed: int = -1) -> None:
        super(iBert, self).__init__(    )
        self.config = {
            "num_labels": 2
        }
        self.model: IBertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained("kssteven/ibert-roberta-base", config=self.config)

    def forward(self, **kwargs) -> torch.Tensor:
        outputs = self.model.forward(**kwargs)
        return outputs.logits

    
