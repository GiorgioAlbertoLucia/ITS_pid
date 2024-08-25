from dataclasses import dataclass

@dataclass
class AxisSpec:

    nbins: int
    xmin: float
    xmax: float
    name: str = ''
    title: str = ''

    @classmethod
    def from_dict(cls, d: dict):
        return cls(d['nbins'], d['xmin'], d['xmax'], d['name'], d['title'])