import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestInputAsOutput(TestCase):
    def test_input_as_output(self):
        class InputAsOutput(torch.nn.Module):
            def forward(self, x, y):
                y_new = y + x + 1
                y_new = y_new * 7
                return (y_new, x, y)

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(InputAsOutput())
        lower_graph_testing(fx_graph, inputs, min_block_size=1)
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"InputAsOutput TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestLoweringPassMembership(TestCase):
    def insert_at_end(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            _aten_lowering_pass,
            _remove_lowering_pass,
        )

        @_aten_lowering_pass
        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[-1])

        _remove_lowering_pass(-1)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)

    def insert_at_index(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            _aten_lowering_pass,
            _remove_lowering_pass,
        )

        @_aten_lowering_pass(index=0)
        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[0])

        _remove_lowering_pass(0)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)


class TestPrimBroadcastFusion(TestCase):
    def test_broadcast_fusion(self):
        class BroadcastFusion(torch.nn.Module):
            def forward(self, x):
                return torch.var_mean(x, keepdim=True)[1]

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(BroadcastFusion())
        expected_ops = {torch.ops.aten.sum.dim_IntList}
        unexpected_ops = {torch.ops.aten.var.default, torch.ops.prims.var.default}

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"BroadcastFusion TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestFP32Accumulation(TestCase):
    def test_fp32_acc(self):
        class FP32Acc(torch.nn.Module):
            def forward(self, input, weight):
                out = torch.ops.aten.mm.default(input, weight)
                return out

        inputs = [
            torch.rand((3, 4)).cuda(),
            torch.rand((4, 5)).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(FP32Acc())
        expected_ops = {torch.ops.aten._to_copy.default, torch.ops.aten.mm.default}
        unexpected_ops = {}

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            use_fp32_acc=True,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()

    def test_fp32_acc_for_addmm(self):
        class FP32Acc(torch.nn.Module):
            def forward(self, input, mat1, mat2):
                out = torch.ops.aten.addmm.default(input, mat1, mat2, beta=20, alpha=2)
                return out

        inputs = [
            torch.rand((3, 5)).cuda(),
            torch.rand((3, 4)).cuda(),
            torch.rand((4, 5)).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(FP32Acc())
        expected_ops = {
            torch.ops.aten._to_copy.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.add.Tensor,
        }
        unexpected_ops = {}

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            use_fp32_acc=True,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()


class TestComplexSubgraph(TestCase):
    def test_complex_subgraph(self):
        BATCH = 1
        SEQ_LEN = 2
        HEADS = 1
        DIM = 2

        class RotaryAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = DIM
                self.wq = torch.nn.Linear(self.dim, self.dim)
                self.seq_len = SEQ_LEN

                self.register_buffer(
                    "freqs_ex_tensor",
                    self._freqs_ex_tensor(),
                    persistent=True,
                )

            def rotary_embedding(self, x, dim, freqs_cis=None):
                x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
                x_out_flatten = torch.view_as_real(x_ * freqs_cis)
                return x_out_flatten.type_as(x)

            def _freqs_ex_tensor(self):
                real = torch.tensor([[[[1.0000]], [[2.0000]]]], device="cuda")
                imag = torch.tensor([[[[0.0000]], [[3.0000]]]], device="cuda")

                z = torch.complex(real, imag)
                return z

            def forward(self, x):
                q = self.wq(x)
                freqs_cis = self._freqs_ex_tensor().to(q.device)
                q_out = self.rotary_embedding(q, self.dim, freqs_cis=freqs_cis)
                return q_out

        inputs = [torch.randn(BATCH, SEQ_LEN, HEADS, DIM).cuda()]
        model = RotaryAttention()
        model = model.cuda()

        expected_ops = {torch.ops.aten.mul.Tensor}
        unexpected_ops = {
            torch.ops.aten.view_as_complex.default,
            torch.ops.aten.view_as_real.default,
        }

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            model,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            model,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs)[0].detach().cpu()
        torch_model_results = model(*inputs)[0].detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"ComplexSubgraph TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
