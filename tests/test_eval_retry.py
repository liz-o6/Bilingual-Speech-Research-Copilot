import unittest
from src.eval.score import should_run

class RetryTests(unittest.TestCase):
    def test_retry_preserves_valid_scores_including_zero(self):
        row={'scores':{'faithfulness':None,'context_recall':0.0},'errors':{'faithfulness':'LLMDidNotFinishException'}}
        self.assertFalse(should_run(row,'faithfulness'))
        self.assertTrue(should_run(row,'faithfulness',True))
        self.assertFalse(should_run(row,'context_recall',True))
        self.assertTrue(should_run(row,'context_precision'))
