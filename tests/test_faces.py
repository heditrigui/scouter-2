import numpy as np

from scouter.faces import FaceDB


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_match_save_load_forget(tmp_path):
    rng = np.random.default_rng(0)
    alex, sam = unit(rng.normal(size=128)), unit(rng.normal(size=128))
    db = FaceDB(tmp_path / "faces.npz")
    assert db.match(alex) == (None, 0.0)

    db.add("Alex", [unit(alex + rng.normal(scale=0.05, size=128)) for _ in range(3)])
    db.add("Sam", [sam])
    db.save()

    db = FaceDB(tmp_path / "faces.npz")
    assert db.people() == {"Alex": 3, "Sam": 1}
    name, sim = db.match(alex)
    assert name == "Alex" and sim > 0.8
    assert db.match(sam)[0] == "Sam"
    assert db.match(unit(rng.normal(size=128)))[0] is None  # a random face matches nobody

    assert db.forget("alex") == 3
    assert db.people() == {"Sam": 1}
    assert db.forget("nobody") == 0
    assert db.forget("Sam") == 1
    assert db.features.shape == (0, 128)
