import pytest

from benchmarks.self_improve.exercises import (
    ExerciseSpec,
    describe_exercise,
    discover_exercises,
    practice_dir,
    select_exercises,
    split_train_val,
)


def test_task_id_and_results_key():
    spec = ExerciseSpec(exercise="wordy", language="python")
    assert spec.task_id == "python/wordy"
    assert spec.results_key == "pi/python/wordy"  # matches aider_polyglot.py:861's key format


def test_practice_dir_matches_lang_descriptors_shape(tmp_path):
    assert practice_dir(tmp_path, "python") == tmp_path / "python" / "exercises" / "practice"


def test_discover_exercises_lists_directories_only(tmp_path):
    (tmp_path / "wordy").mkdir()
    (tmp_path / "bowling").mkdir()
    (tmp_path / "not_a_dir.txt").write_text("x")
    assert discover_exercises(tmp_path) == ["bowling", "wordy"]


def test_discover_exercises_handles_missing_dir(tmp_path):
    assert discover_exercises(tmp_path / "does-not-exist") == []


def test_select_exercises_is_deterministic_given_seed():
    available = ["a", "b", "c", "d", "e"]
    first = select_exercises(available, count=3, seed=42)
    second = select_exercises(available, count=3, seed=42)
    assert first == second


def test_select_exercises_different_seeds_can_differ():
    available = ["a", "b", "c", "d", "e", "f", "g", "h"]
    a = select_exercises(available, count=3, seed=1)
    b = select_exercises(available, count=3, seed=2)
    assert a != b


def test_select_exercises_explicit_list_bypasses_sampling():
    available = ["a", "b", "c"]
    chosen = select_exercises(available, count=99, seed=1, explicit=["b", "a"])
    assert [s.exercise for s in chosen] == ["b", "a"]


def test_select_exercises_explicit_unknown_name_raises_with_available_list():
    with pytest.raises(ValueError, match=r"unknown exercise.*typo.*available.*\['a', 'b'\]"):
        select_exercises(["a", "b"], count=1, seed=1, explicit=["typo"])


def test_select_exercises_explicit_duplicate_name_raises():
    """A repeated --exercises name would otherwise let split_train_val place
    the same task in both train and val."""
    with pytest.raises(ValueError, match="duplicate exercise"):
        select_exercises(["a", "b"], count=1, seed=1, explicit=["a", "a"])


def test_select_exercises_raises_when_count_exceeds_available():
    with pytest.raises(ValueError, match="only 2 available"):
        select_exercises(["a", "b"], count=5, seed=1)


def test_select_exercises_respects_exclude():
    chosen = select_exercises(["a", "b", "c"], count=2, seed=1, exclude=("a",))
    assert "a" not in {s.exercise for s in chosen}


def test_split_train_val_is_deterministic_given_seed():
    specs = [ExerciseSpec(exercise=f"e{i}") for i in range(6)]
    t1, v1 = split_train_val(specs, val_count=2, seed=7)
    t2, v2 = split_train_val(specs, val_count=2, seed=7)
    assert t1 == t2 and v1 == v2


def test_split_train_val_disjoint_and_covers_all():
    specs = [ExerciseSpec(exercise=f"e{i}") for i in range(6)]
    train, val = split_train_val(specs, val_count=2, seed=1)
    assert len(val) == 2
    assert len(train) == 4
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(specs)


def test_split_train_val_rejects_val_count_out_of_range():
    specs = [ExerciseSpec(exercise=f"e{i}") for i in range(3)]
    with pytest.raises(ValueError):
        split_train_val(specs, val_count=0, seed=1)
    with pytest.raises(ValueError):
        split_train_val(specs, val_count=3, seed=1)


def test_describe_exercise_prefers_instructions_md(tmp_path):
    ex_dir = tmp_path / "wordy" / ".docs"
    ex_dir.mkdir(parents=True)
    (ex_dir / "instructions.md").write_text("Real instructions.")
    (tmp_path / "wordy" / "README.md").write_text("Should not be used.")
    desc = describe_exercise(ExerciseSpec(exercise="wordy"), tmp_path)
    assert desc == "Real instructions."


def test_describe_exercise_falls_back_to_readme(tmp_path):
    (tmp_path / "wordy").mkdir()
    (tmp_path / "wordy" / "README.md").write_text("From readme.")
    desc = describe_exercise(ExerciseSpec(exercise="wordy"), tmp_path)
    assert desc == "From readme."


def test_describe_exercise_falls_back_to_stub_and_test_listing(tmp_path):
    ex_dir = tmp_path / "wordy"
    ex_dir.mkdir()
    (ex_dir / "wordy.py").write_text("def solve(): pass\n")
    (ex_dir / "wordy_test.py").write_text("\n".join(f"line {i}" for i in range(50)))
    desc = describe_exercise(ExerciseSpec(exercise="wordy"), tmp_path)
    assert "wordy.py" in desc
    assert "wordy_test.py" in desc
    assert "line 0" in desc
    assert "line 49" not in desc  # capped at the first 40 lines


def test_describe_exercise_handles_missing_directory(tmp_path):
    desc = describe_exercise(ExerciseSpec(exercise="does-not-exist"), tmp_path)
    assert "not found" in desc
