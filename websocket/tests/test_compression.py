import hashlib
import struct
import unittest
import zlib
from base64 import encodebytes as base64encode

import websocket as ws
from websocket._abnf import ABNF
from websocket._compression import PerMessageDeflate, deflate_session
from websocket._exceptions import (
    WebSocketException,
    WebSocketPayloadException,
    WebSocketProtocolException,
)
from websocket._handshake import _get_handshake_headers, handshake

"""
test_compression.py
websocket - WebSocket client library for Python

Copyright 2026 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

MASK_KEY = b"\x01\x02\x03\x04"


class SockMock:
    def __init__(self, data=b""):
        self.data = data
        self.sent = []

    def gettimeout(self):
        return None

    def recv(self, bufsize):
        chunk, self.data = self.data[:bufsize], self.data[bufsize:]
        return chunk

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def close(self):
        pass


def server_frame(opcode, payload, fin=1, rsv1=0):
    header = bytes([fin << 7 | rsv1 << 6 | opcode])
    if len(payload) < 126:
        return header + bytes([len(payload)]) + payload
    return header + bytes([126]) + struct.pack("!H", len(payload)) + payload


def server_compress(compressor, data):
    compressed = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
    return compressed[:-4]


def new_compressor():
    return zlib.compressobj(wbits=-zlib.MAX_WBITS)


def parse_sent(data):
    """Split one masked client frame into (rsv1, opcode, payload)."""
    rsv1 = data[0] >> 6 & 1
    opcode = data[0] & 0xF
    length = data[1] & 0x7F
    offset = 2
    if length == 126:
        length = struct.unpack("!H", data[2:4])[0]
        offset = 4
    mask = data[offset : offset + 4]
    payload = ABNF.mask(mask, data[offset + 4 : offset + 4 + length])
    return rsv1, opcode, payload


def connected_socket(data=b"", session=None, **options):
    sock = ws.WebSocket(**options)
    sock.set_mask_key(lambda _: MASK_KEY)
    sock.sock = SockMock(data)
    sock.connected = True
    if session is not None:
        sock.deflate = session
        sock.frame_buffer.rsv1_allowed = True
    return sock


class OfferTest(unittest.TestCase):
    def test_default_offer(self):
        self.assertEqual(
            PerMessageDeflate().offer(), "permessage-deflate; client_max_window_bits"
        )

    def test_offer_with_every_parameter(self):
        offer = PerMessageDeflate(
            client_no_context_takeover=True,
            server_no_context_takeover=True,
            client_max_window_bits=10,
            server_max_window_bits=12,
        ).offer()
        self.assertEqual(
            offer,
            "permessage-deflate; client_no_context_takeover; server_no_context_takeover; "
            "client_max_window_bits=10; server_max_window_bits=12",
        )

    def test_offer_without_client_max_window_bits(self):
        self.assertEqual(
            PerMessageDeflate(client_max_window_bits=False).offer(),
            "permessage-deflate",
        )

    def test_invalid_settings(self):
        for kwargs in (
            {"client_max_window_bits": 8},
            {"client_max_window_bits": 16},
            {"server_max_window_bits": 7},
            {"server_max_window_bits": 16},
            {"compress_level": 10},
            {"max_message_size": -1},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertRaises(ValueError, PerMessageDeflate, **kwargs)


class AcceptTest(unittest.TestCase):
    def test_declined(self):
        self.assertIsNone(PerMessageDeflate().accept(None))
        self.assertIsNone(PerMessageDeflate().accept(""))

    def test_accepted(self):
        session = PerMessageDeflate().accept("permessage-deflate")
        self.assertIsInstance(session, deflate_session)
        self.assertFalse(session.server_no_context_takeover)
        self.assertFalse(session.client_no_context_takeover)
        self.assertEqual(session.client_window_bits, 15)

    def test_accepted_with_parameters(self):
        session = PerMessageDeflate().accept(
            "Permessage-Deflate; server_no_context_takeover; client_no_context_takeover; "
            'server_max_window_bits=10; client_max_window_bits="11"'
        )
        self.assertTrue(session.server_no_context_takeover)
        self.assertTrue(session.client_no_context_takeover)
        self.assertEqual(session.client_window_bits, 11)

    def test_offered_client_window_bits_cap_the_compressor(self):
        session = PerMessageDeflate(client_max_window_bits=10).accept(
            "permessage-deflate; client_max_window_bits=12"
        )
        self.assertEqual(session.client_window_bits, 10)

    def test_offered_client_no_context_takeover_is_kept(self):
        session = PerMessageDeflate(client_no_context_takeover=True).accept(
            "permessage-deflate"
        )
        self.assertTrue(session.client_no_context_takeover)

    def test_invalid_responses(self):
        for response in (
            "permessage-deflate, permessage-deflate",
            "x-webkit-deflate-frame",
            "permessage-deflate; server_no_context_takeover; server_no_context_takeover",
            "permessage-deflate; unknown_parameter",
            "permessage-deflate; server_no_context_takeover=1",
            "permessage-deflate; client_no_context_takeover=1",
            "permessage-deflate; server_max_window_bits",
            "permessage-deflate; server_max_window_bits=7",
            "permessage-deflate; server_max_window_bits=16",
            "permessage-deflate; server_max_window_bits=010",
            "permessage-deflate; server_max_window_bits=ten",
            "permessage-deflate; client_max_window_bits",
            "permessage-deflate; client_max_window_bits=8",
        ):
            with self.subTest(response=response):
                self.assertRaises(
                    WebSocketException, PerMessageDeflate().accept, response
                )

    def test_server_window_bits_above_the_offer(self):
        self.assertRaises(
            WebSocketException,
            PerMessageDeflate(server_max_window_bits=10).accept,
            "permessage-deflate; server_max_window_bits=11",
        )

    def test_unoffered_client_window_bits(self):
        self.assertRaises(
            WebSocketException,
            PerMessageDeflate(client_max_window_bits=False).accept,
            "permessage-deflate; client_max_window_bits=15",
        )


class HandshakeTest(unittest.TestCase):
    KEY = "dGhlIHNhbXBsZSBub25jZQ=="

    def response(self, extensions=None):
        accept = base64encode(
            hashlib.sha1(
                f"{self.KEY}258EAFA5-E914-47DA-95CA-C5AB0DC85B11".encode()
            ).digest()
        ).strip()
        lines = [
            b"HTTP/1.1 101 Switching Protocols",
            b"Upgrade: websocket",
            b"Connection: Upgrade",
            b"Sec-WebSocket-Accept: " + accept,
        ]
        for extension in extensions or ():
            lines.append(b"Sec-WebSocket-Extensions: " + extension.encode())
        return b"\r\n".join(lines) + b"\r\n\r\n"

    def do_handshake(self, extensions=None, **options):
        sock = SockMock(self.response(extensions))
        return handshake(
            sock,
            "ws://example.com/",
            "example.com",
            80,
            "/",
            header={"Sec-WebSocket-Key": self.KEY},
            **options,
        )

    def test_offer_header(self):
        headers, _ = _get_handshake_headers(
            "/", "ws://example.com", "example.com", 80, {"compression": True}
        )
        self.assertIn(
            "Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits",
            headers,
        )

    def test_no_offer_by_default(self):
        headers, _ = _get_handshake_headers(
            "/", "ws://example.com", "example.com", 80, {}
        )
        self.assertFalse(any(h.startswith("Sec-WebSocket-Extensions") for h in headers))

    def test_offer_conflicts_with_a_custom_extensions_header(self):
        self.assertRaises(
            WebSocketException,
            _get_handshake_headers,
            "/",
            "ws://example.com",
            "example.com",
            80,
            {
                "compression": True,
                "header": ["Sec-WebSocket-Extensions: permessage-deflate"],
            },
        )

    def test_invalid_compression_option(self):
        self.assertRaises(
            WebSocketException,
            _get_handshake_headers,
            "/",
            "ws://example.com",
            "example.com",
            80,
            {"compression": "yes"},
        )

    def test_accepted(self):
        response = self.do_handshake(["permessage-deflate"], compression=True)
        self.assertIsInstance(response.deflate, deflate_session)

    def test_declined(self):
        self.assertIsNone(self.do_handshake(compression=True).deflate)

    def test_extension_header_split_over_lines(self):
        self.assertRaises(
            WebSocketException,
            self.do_handshake,
            ["permessage-deflate", "permessage-deflate"],
            compression=True,
        )

    def test_extensions_ignored_without_the_option(self):
        self.assertIsNone(self.do_handshake(["permessage-deflate"]).deflate)


class ReceiveTest(unittest.TestCase):
    def session(self, response="permessage-deflate", **settings):
        return PerMessageDeflate(**settings).accept(response)

    def test_messages_share_one_context(self):
        compressor = new_compressor()
        message = b"Hello, Hello, Hello"
        data = server_frame(
            ABNF.OPCODE_TEXT, server_compress(compressor, message), rsv1=1
        )
        second = server_compress(compressor, message)
        data += server_frame(ABNF.OPCODE_TEXT, second, rsv1=1)
        sock = connected_socket(data, self.session())
        self.assertEqual(sock.recv(), "Hello, Hello, Hello")
        self.assertEqual(sock.recv(), "Hello, Hello, Hello")
        self.assertLess(len(second), len(message))

    def test_fragmented_message_with_a_control_frame(self):
        message = b"fragmented " * 30
        compressed = server_compress(new_compressor(), message)
        half = len(compressed) // 2
        data = server_frame(ABNF.OPCODE_BINARY, compressed[:half], fin=0, rsv1=1)
        data += server_frame(ABNF.OPCODE_PING, b"")
        data += server_frame(ABNF.OPCODE_CONT, compressed[half:])
        sock = connected_socket(data, self.session())
        self.assertEqual(sock.recv_data(), (ABNF.OPCODE_BINARY, message))

    def test_fire_cont_frame_receives_each_fragment_decompressed(self):
        message = b"0123456789" * 20
        compressed = server_compress(new_compressor(), message)
        data = server_frame(ABNF.OPCODE_BINARY, compressed[:5], fin=0, rsv1=1)
        data += server_frame(ABNF.OPCODE_CONT, compressed[5:])
        sock = connected_socket(data, self.session(), fire_cont_frame=True)
        _, first = sock.recv_data_frame()
        _, second = sock.recv_data_frame()
        self.assertEqual(first.data + second.data, message)

    def test_uncompressed_messages_between_compressed_ones(self):
        compressor = new_compressor()
        data = server_frame(ABNF.OPCODE_BINARY, b"plain")
        data += server_frame(
            ABNF.OPCODE_BINARY, server_compress(compressor, b"packed"), rsv1=1
        )
        data += server_frame(ABNF.OPCODE_BINARY, b"plain again")
        sock = connected_socket(data, self.session())
        self.assertEqual(sock.recv(), b"plain")
        self.assertEqual(sock.recv(), b"packed")
        self.assertEqual(sock.recv(), b"plain again")

    def test_server_no_context_takeover(self):
        data = b""
        for _ in range(2):
            data += server_frame(
                ABNF.OPCODE_TEXT, server_compress(new_compressor(), b"same"), rsv1=1
            )
        sock = connected_socket(
            data, self.session("permessage-deflate; server_no_context_takeover")
        )
        self.assertEqual(sock.recv(), "same")
        self.assertEqual(sock.recv(), "same")

    def test_rsv1_on_a_continuation_frame(self):
        compressed = server_compress(new_compressor(), b"split message")
        data = server_frame(ABNF.OPCODE_TEXT, compressed[:3], fin=0, rsv1=1)
        data += server_frame(ABNF.OPCODE_CONT, compressed[3:], rsv1=1)
        sock = connected_socket(data, self.session())
        self.assertRaises(WebSocketProtocolException, sock.recv)

    def test_rsv1_on_a_control_frame(self):
        sock = connected_socket(
            server_frame(ABNF.OPCODE_PING, b"", rsv1=1), self.session()
        )
        self.assertRaises(WebSocketProtocolException, sock.recv)

    def test_rsv1_without_the_extension(self):
        data = server_frame(
            ABNF.OPCODE_TEXT, server_compress(new_compressor(), b"packed"), rsv1=1
        )
        sock = connected_socket(data)
        self.assertRaises(WebSocketProtocolException, sock.recv)

    def test_invalid_utf8_after_decompression(self):
        data = server_frame(
            ABNF.OPCODE_TEXT, server_compress(new_compressor(), b"\xffbad"), rsv1=1
        )
        sock = connected_socket(data, self.session())
        self.assertRaises(WebSocketPayloadException, sock.recv)

    def test_invalid_compressed_data(self):
        data = server_frame(ABNF.OPCODE_BINARY, b"\xff\xff\xff\xff", rsv1=1)
        sock = connected_socket(data, self.session())
        self.assertRaises(WebSocketProtocolException, sock.recv)

    def test_max_message_size(self):
        data = server_frame(
            ABNF.OPCODE_BINARY, server_compress(new_compressor(), b"y" * 101), rsv1=1
        )
        sock = connected_socket(data, self.session(max_message_size=100))
        self.assertRaises(WebSocketPayloadException, sock.recv)

    def test_max_message_size_counts_every_fragment(self):
        compressed = server_compress(new_compressor(), b"z" * 101)
        data = server_frame(ABNF.OPCODE_BINARY, compressed[:2], fin=0, rsv1=1)
        data += server_frame(ABNF.OPCODE_CONT, compressed[2:])
        sock = connected_socket(data, self.session(max_message_size=100))
        self.assertRaises(WebSocketPayloadException, sock.recv)

    def test_no_max_message_size(self):
        message = b"q" * 100000
        compressed = server_compress(new_compressor(), message)
        sock = connected_socket(
            server_frame(ABNF.OPCODE_BINARY, compressed, rsv1=1),
            self.session(max_message_size=None),
        )
        self.assertEqual(sock.recv(), message)


class SendTest(unittest.TestCase):
    def sent_messages(self, sock):
        return [parse_sent(data) for data in sock.sock.sent]

    def test_messages_are_compressed_with_one_context(self):
        sock = connected_socket(
            session=PerMessageDeflate().accept("permessage-deflate")
        )
        sock.send("Hello, Hello, Hello")
        sock.send("Hello, Hello, Hello")
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        (rsv1_a, op_a, first), (rsv1_b, op_b, second) = self.sent_messages(sock)
        self.assertEqual((rsv1_a, op_a, rsv1_b, op_b), (1, 1, 1, 1))
        self.assertLess(len(second), len(first))
        for payload in (first, second):
            self.assertEqual(
                decompressor.decompress(payload + b"\x00\x00\xff\xff"),
                b"Hello, Hello, Hello",
            )

    def test_client_no_context_takeover(self):
        sock = connected_socket(
            session=PerMessageDeflate().accept(
                "permessage-deflate; client_no_context_takeover"
            )
        )
        sock.send_binary(b"Hello, Hello, Hello")
        sock.send_binary(b"Hello, Hello, Hello")
        (_, _, first), (_, op, second) = self.sent_messages(sock)
        self.assertEqual(op, ABNF.OPCODE_BINARY)
        self.assertEqual(first, second)
        self.assertEqual(
            zlib.decompressobj(-zlib.MAX_WBITS).decompress(
                second + b"\x00\x00\xff\xff"
            ),
            b"Hello, Hello, Hello",
        )

    def test_client_window_bits(self):
        sock = connected_socket(
            session=PerMessageDeflate().accept(
                "permessage-deflate; client_max_window_bits=9"
            )
        )
        message = bytes(range(256)) * 8
        sock.send_binary(message)
        (_, _, payload) = self.sent_messages(sock)[0]
        self.assertEqual(
            zlib.decompressobj(-9).decompress(payload + b"\x00\x00\xff\xff"), message
        )

    def test_empty_message(self):
        sock = connected_socket(
            session=PerMessageDeflate().accept("permessage-deflate")
        )
        sock.send("")
        (rsv1, _, payload) = self.sent_messages(sock)[0]
        self.assertEqual(rsv1, 1)
        self.assertEqual(
            zlib.decompressobj(-zlib.MAX_WBITS).decompress(
                payload + b"\x00\x00\xff\xff"
            ),
            b"",
        )

    def test_control_frames_and_raw_frames_are_not_compressed(self):
        sock = connected_socket(
            session=PerMessageDeflate().accept("permessage-deflate")
        )
        sock.ping("ping")
        sock.send_frame(ABNF.create_frame("raw", ABNF.OPCODE_TEXT))
        self.assertEqual(
            self.sent_messages(sock),
            [(0, ABNF.OPCODE_PING, b"ping"), (0, ABNF.OPCODE_TEXT, b"raw")],
        )

    def test_without_the_extension(self):
        sock = connected_socket()
        sock.send("plain")
        self.assertEqual(self.sent_messages(sock), [(0, ABNF.OPCODE_TEXT, b"plain")])


class AppTest(unittest.TestCase):
    def test_compression_is_passed_to_connect(self):
        options = []

        def connect(self, url, **kwargs):
            options.append(kwargs)
            raise WebSocketException("stop")

        original = ws.WebSocket.connect
        ws.WebSocket.connect = connect
        try:
            compression = PerMessageDeflate()
            app = ws.WebSocketApp("ws://example.com", compression=compression)
            app.run_forever()
        finally:
            ws.WebSocket.connect = original
        self.assertIs(options[0]["compression"], compression)


if __name__ == "__main__":
    unittest.main()
