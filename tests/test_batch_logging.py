import logging
import builtins
from cellfinder.core.classify.classify import BatchEndCallback

def test_batch_end_callback_invokes_user_function():
    called_batches = []
    # Define a simple callback function that appends the batch number to our list
    def record_batch(batch_number):
        called_batches.append(batch_number)
    # Wrap it in BatchEndCallback
    cb = BatchEndCallback(record_batch)
    # Simulate the end of batch #5
    cb.on_predict_batch_end(batch=5)
    # Now, our list should contain 5
    assert called_batches == [5]
