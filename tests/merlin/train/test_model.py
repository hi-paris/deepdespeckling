#!/usr/bin/env python
"""Tests for `merlin` package."""

import sys
sys.path.insert(0, 'C:/Users/ykemiche/OneDrive - Capgemini/Desktop/deepdespeckling/deepdespeckling')

from deepdespeckling.merlin.train.model import *
import numpy as np
import torch 


batch_size=1
val_batch_size=1
device="cuda:0" if torch.cuda.is_available() else "cpu"

model = Model(batch_size,val_batch_size,device)
x=np.random.rand(1,256,256,1)
x=torch.tensor(x)
x=x.type(torch.float32)

y=np.random.rand(1,256,256,1)
y=torch.tensor(y)
y=y.type(torch.float32)

out=model.forward(x,1)




class TestNet():
    """Test class for Net1 function"""

    def test_net_instance_good_shape(self):
        """Test that a Net1 instance has good shape
        Parameters
        ----------
        capfd: fixture
        Allows access to stdout/stderr output created
        during test execution.
        Returns
        -------
        None
        """
         
        assert out.shape==torch.Size([1, 1, 256, 256])

    def test_net_forward_returns_good_type(self, ):
        """Tests that a net1 instance has good type
        Returns
        -------
        None
        """ 

    assert type(out) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_loss_function_good_shape(self):
        """Test that a Net1 instance has good shape
        Parameters
        ----------
        capfd: fixture
        Allows access to stdout/stderr output created
        during test execution.
        Returns
        -------
        None
        """
        loss=model.loss_function(out,y,1)
        print(loss.shape)
        assert type(loss) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_loss_function_returns_good_type(self, ):
        """Tests that a net1 instance has good type
        Returns
        -------
        None
        """ 
        loss=model.loss_function(out,y,1)

        assert loss.shape == torch.Size([])

 