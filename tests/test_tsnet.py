"""Tests for TSF-TimesNet (atsf_dcg/model_tsnet.py, plan v6 Stage 2b).

Covers:
- forward shapes for every config of the replication cell, BOTH archs
  (ATSFDCG / TSFTimesNet) and BOTH datasets' channel counts (NPPAD 96 /
  TEP 52): logits (B,K), cache contract (H_t/H_f/alpha/G/z/logits_t/f);
- ``_last_alpha`` captured for adaptive fusion;
- fixed_global / fixed_class init validation errors mirror ATSFDCG
  (inherited constructor contract);
- the vanilla baseline (tsnet_vanilla: use_spectral=False, fusion="none",
  gating="none") forwards as a plain TimesNet classifier;
- gradient flows through the TimesNet temporal branch (backward on a
  batch; TimesBlock convs receive grads; both branches get grads under
  the full config);
- the OGM (R1) aux-head path trains one epoch on synthetic micro data for
  full_r1 under BOTH archs via the real ``train_one`` entry point.

Run from the project root:  ``python tests/test_tsnet.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from atsf_dcg.data import make_synthetic  # noqa: E402
from atsf_dcg.model import ATSFDCG  # noqa: E402
from atsf_dcg.model_tsnet import TimesNetTemporalBranch, TSFTimesNet  # noqa: E402
from atsf_dcg.run_experiments import build_configs  # noqa: E402
from atsf_dcg.train import train_one  # noqa: E402
from atsf_dcg.utils import ExpConfig, set_seed  # noqa: E402

K = 6  # classes in all shape checks


def _forward_shape_checks() -> None:
    set_seed(0)
    for n_channels in (96, 52):
        x = torch.randn(4, n_channels, 128)
        for arch, cls in (("atsf", ATSFDCG), ("tsnet", TSFTimesNet)):
            for cfg in build_configs(cell="replication", arch=arch):
                if cfg.name == "tsnet_vanilla":
                    assert arch == "tsnet"  # Cell D config only exists here
                model = cls(n_channels, K, cfg)
                logits, cache = model(x)
                assert logits.shape == (4, K), (arch, cfg.name, logits.shape)
                assert cache["z"].shape == (4, 128)
                # branch tensors
                if cfg.use_temporal:
                    assert cache["H_t"].shape == (4, 128, 64), \
                        (arch, cfg.name, cache["H_t"].shape)
                else:
                    assert cache["H_t"] is None
                if cfg.use_spectral:
                    assert cache["H_f"].shape == (4, 128, 64)
                else:
                    assert cache["H_f"] is None
                # fusion cache
                if cfg.use_temporal and cfg.use_spectral \
                        and cfg.fusion == "adaptive":
                    assert cache["alpha"] is not None
                    assert model._last_alpha is cache["alpha"]
                    assert bool((cache["alpha"] > 0).all()) \
                        and bool((cache["alpha"] < 1).all())
                elif cfg.fusion in ("none",) or not (
                        cfg.use_temporal and cfg.use_spectral):
                    assert cache["alpha"] is None
                # gating cache
                if cfg.gating == "none":
                    assert cache["G"] is None
                else:
                    assert cache["G"].shape == (4, 128, 64)
                # R1 aux heads
                if cfg.r1_balanced:
                    if cfg.use_temporal:
                        assert cache["logits_t"].shape == (4, K)
                    if cfg.use_spectral:
                        assert cache["logits_f"].shape == (4, K)
        print(f"[test] OK: forward shapes, {n_channels} channels, both "
              "archs, all replication configs.")

    # tsnet_vanilla is exactly the plain TimesNet classifier: no fusion /
    # gating / spectral modules at all
    v = TSFTimesNet(52, K, ExpConfig(name="tsnet_vanilla",
                                     use_spectral=False, fusion="none",
                                     gating="none"))
    assert isinstance(v.temporal_branch, TimesNetTemporalBranch)
    for attr in ("spectral_branch", "alpha_head", "fuse_conv", "gate_conv",
                 "gate"):
        assert not hasattr(v, attr), attr
    logits, cache = v(torch.randn(2, 52, 128))
    assert logits.shape == (2, K) and cache["alpha"] is None \
        and cache["G"] is None and cache["H_f"] is None
    print("[test] OK: tsnet_vanilla = plain TimesNet classifier (no "
          "spectral/fusion/gating modules).")


def _fixed_fusion_validation_checks() -> None:
    """TSFTimesNet init validation mirrors ATSFDCG (inherited contract)."""
    for cls in (ATSFDCG, TSFTimesNet):
        for kw in (dict(fusion="fixed_global"),
                   dict(fusion="fixed_global", fusion_fixed_alpha=1.5),
                   dict(fusion="fixed_global", fusion_fixed_alpha=0.5,
                        use_spectral=False),
                   dict(fusion="fixed_class"),
                   dict(fusion="fixed_class",
                        fusion_fixed_class_alpha=(0.1, 0.2)),
                   dict(fusion="fixed_class",
                        fusion_fixed_class_alpha=(0.1,) * (K - 1) + (-0.1,))):
            try:
                cls(52, K, ExpConfig(name="bad", **kw))
                raise AssertionError(f"{cls.__name__} accepted {kw}")
            except ValueError:
                pass
    # fixed modes work end-to-end on tsnet too (constant alpha cache,
    # oracle y required for fixed_class)
    m = TSFTimesNet(52, K, ExpConfig(name="fg", fusion="fixed_global",
                                     fusion_fixed_alpha=0.3))
    _, cache = m(torch.randn(3, 52, 128))
    assert torch.allclose(cache["alpha"], torch.full((3, 1, 64), 0.3))
    m = TSFTimesNet(52, K, ExpConfig(
        name="fc", fusion="fixed_class",
        fusion_fixed_class_alpha=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
    try:
        m(torch.randn(3, 52, 128))
        raise AssertionError("fixed_class accepted y=None")
    except ValueError as e:
        assert "true labels" in str(e)
    y = torch.tensor([0, 1, 2])
    _, cache = m(torch.randn(3, 52, 128), y)
    assert torch.allclose(cache["alpha"][:, 0, 0],
                          torch.tensor([0.1, 0.2, 0.3]))
    print("[test] OK: fixed_global/fixed_class validation + oracle "
          "contract mirror ATSFDCG.")


def _gradient_checks() -> None:
    set_seed(0)
    x = torch.randn(4, 52, 128)
    y = torch.randint(0, K, (4,))
    model = TSFTimesNet(52, K, ExpConfig(name="full"))
    model.train()
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    # TimesNet temporal branch receives gradients
    tb = model.temporal_branch
    assert tb.blocks[0].conv.convs[0].weight.grad is not None
    assert tb.blocks[1].conv.expand.weight.grad is not None
    assert tb.proj[0][0].weight.grad is not None  # _conv_relu_bn conv
    # both branches + shared head receive gradients (rho path defined)
    assert model.spectral_branch[0][0].weight.grad is not None
    assert model.alpha_head.weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    tp, sp = model.temporal_params(), model.spectral_params()
    assert tp and sp
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in tp)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in sp)
    # vanilla: balance_loss is scalar 0, grads confined to temporal branch
    v = TSFTimesNet(52, K, ExpConfig(name="tsnet_vanilla",
                                     use_spectral=False, fusion="none",
                                     gating="none"))
    logits, _ = v(x)
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert float(v.balance_loss()) == 0.0
    assert v.spectral_params() == []
    print("[test] OK: gradients flow through TimesBlocks, both branches, "
          "fusion head and BiLSTM head; vanilla balance_loss == 0.")


def _ogm_training_checks() -> None:
    """R1 (OGM aux heads): one train_one epoch under BOTH archs."""
    from dataclasses import replace

    bundle = make_synthetic(n_classes=6, runs_per_class=3, n_channels=16,
                            seed=0)
    for arch in ("atsf", "tsnet"):
        cfg = replace(ExpConfig(name="full_r1", r1_balanced=True),
                      epochs=1, patience=1, seed=42)
        res = train_one(cfg, bundle, arch=arch)
        rec = res["diagnostics"][0]
        for key in ("acc_t_aux", "acc_f_aux", "coef_t", "coef_f"):
            assert key in rec, (arch, key)
        assert 0.0 <= rec["acc_t_aux"] <= 1.0
        assert 0.0 <= rec["acc_f_aux"] <= 1.0
        assert res["epochs_run"] == 1
        assert 0.0 <= res["accuracy"] <= 1.0
        print(f"[test] OK: full_r1 OGM one-epoch train under arch={arch} "
              f"(acc={res['accuracy']:.4f}, coef_t={rec['coef_t']:.3f}, "
              f"coef_f={rec['coef_f']:.3f}).")


def main() -> None:
    t0 = time.time()
    _forward_shape_checks()
    _fixed_fusion_validation_checks()
    _gradient_checks()
    _ogm_training_checks()
    print(f"[test_tsnet] ALL OK ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
