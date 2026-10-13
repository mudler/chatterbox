"""Exercise checkpoint selection without downloading weights or importing torch."""
import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1] / "src/chatterbox"


class TurboLoadingTests(unittest.TestCase):
    def setUp(self):
        config_globals = {}
        exec((ROOT / 'models/turbo/llama_configs.py').read_text(), config_globals)
        tree = ast.parse((ROOT / 'models/turbo/t3_config.py').read_text())
        tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)]
        exec(compile(tree, 't3_config.py', 'exec'), config_globals)
        self.configs = config_globals['LLAMA_CONFIGS']
        self.globals = dict(
            Path=Path, T3Config=config_globals['T3Config'], T3=MagicMock(),
            S3Gen=MagicMock(), VoiceEncoder=MagicMock(), EnTokenizer=object,
            Conditionals=MagicMock(), AutoTokenizer=MagicMock(),
            load_file=MagicMock(return_value={}), torch=MagicMock(),
            perth=MagicMock(), S3_SR=16000, S3GEN_SR=24000,
        )
        self.globals['AutoTokenizer'].from_pretrained.return_value.__len__.return_value = 50276
        tree = ast.parse((ROOT / 'tts_turbo.py').read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'ChatterboxTurboTTS']
        exec(compile(tree, 'tts_turbo.py', 'exec'), self.globals)

    def test_turbo_checkpoint_and_architecture(self):
        self.check_variant(False, 'GPT2_medium', 1024, 't3_turbo_v1.safetensors')

    def test_nano_checkpoint_and_architecture(self):
        self.check_variant(True, 'GPT2_small', 768, 't3_nano_v1.safetensors')

    def check_variant(self, nano, architecture, channels, checkpoint):
        model = self.globals['ChatterboxTurboTTS'].from_local('/missing/model', 'cpu', nano=nano)
        hp = self.globals['T3'].call_args.args[0]
        self.assertEqual(hp.llama_config_name, architecture)
        self.assertEqual(hp.n_channels, channels)
        self.assertEqual(hp.text_tokens_dict_size, 50276)
        self.assertEqual(hp.speech_tokens_dict_size, 6563)
        self.assertFalse(hp.use_perceiver_resampler)
        self.assertFalse(hp.emotion_adv)
        self.assertIsNone(hp.input_pos_emb)
        self.globals['S3Gen'].assert_called_once_with(meanflow=True)
        loaded = [call.args[0].name for call in self.globals['load_file'].call_args_list]
        self.assertEqual(loaded, ['ve.safetensors', checkpoint, 's3gen_meanflow.safetensors'])
        self.assertEqual(model.model_label, 'Nano' if nano else 'Turbo')
        self.assertEqual(model.sr, 24000)


if __name__ == '__main__':
    unittest.main()
