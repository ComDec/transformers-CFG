import logging
from unittest import TestCase

from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer


class TestUnicode(TestCase):
    def test_accept_japanese(self):
        """
        Test that we can accept japanese characters
        """

        japanese = "こんにちは世界"
        with open("examples/grammars/japanese.ebnf") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)

        start_rule_id = parsed_grammar.symbol_table["root"]

        recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

        bytes_japanese = bytes(japanese, "utf-8")
        logging.debug(f"bytes_japanese: {bytes_japanese} of length {len(bytes_japanese)}")
        # こんにちは世界

        head_bytes = bytes_japanese[:8]
        # partial_utf8 = PartialUTF8()
        parsing_state = recognizer._update_state_with_bytes(head_bytes)

        # non empty stack means that the bytes were accepted
        self.assertTrue(len(parsing_state.stacks) > 0)

    def test_accept_japanese_progressive(self):
        #######################
        # Now consider the case of progressive matching
        #######################

        japanese = "こんにちは世界"
        with open("examples/grammars/japanese.ebnf") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)

        start_rule_id = parsed_grammar.symbol_table["root"]

        recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

        bytes_japanese = bytes(japanese, "utf-8")
        logging.debug(f"bytes_japanese: {bytes_japanese} of length {len(bytes_japanese)}")

        byte_tokens = [bytes_japanese[i] for i in range(len(bytes_japanese))]
        # cast into bytes
        byte_tokens = [bytes([byte]) for byte in byte_tokens]

        parsing_state = recognizer.get_initial_parsing_state()

        # parsing_state = recognizer.init_parsing_state
        for i, byte in enumerate(byte_tokens):
            parsing_state = recognizer._update_state_with_bytes(byte, parsing_state)
            self.assertTrue(len(parsing_state.stacks) > 0)

    def test_accept_emoji(self):
        """
        Test that we can accept emoji
        """

        emoji = "😀😄😂"
        with open("examples/grammars/emoji.ebnf") as file:
            input_text = file.read()
        parsed_grammar = parse_ebnf(input_text)

        start_rule_id = parsed_grammar.symbol_table["root"]

        recognizer = StringRecognizer(parsed_grammar.grammar_encoding, start_rule_id)

        bytes_emoji = bytes(emoji, "utf-8")
        logging.debug(f"bytes_emoji: {bytes_emoji} of length {len(bytes_emoji)}")
        # 😀😄😂

        # partial_utf8 = PartialUTF8()
        parsing_state = recognizer._update_state_with_bytes(bytes_emoji)
        # non empty stack means that the bytes were accepted
        self.assertTrue(len(parsing_state.stacks) > 0)
