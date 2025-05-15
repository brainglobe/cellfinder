from dataclasses import asdict, dataclass

from cellfinder.napari.input_container import asdict_no_copy


def test_asdict_no_copy():
    @dataclass
    class Data:
        a: int = 5
        b: str = "hello"

    data = Data(a=12, b="bye")
    assert asdict_no_copy(data) == asdict(data)
