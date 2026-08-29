from atomspike.convert.pmsm import convert_checkpoint, enable_pmsm, pmsm_status

__all__ = ["convert_checkpoint", "convert_spiked_attention", "enable_pmsm", "pmsm_status"]


def __getattr__(name):
    if name == "convert_spiked_attention":
        from atomspike.convert.spiked_attention import convert_spiked_attention

        return convert_spiked_attention
    raise AttributeError(name)
