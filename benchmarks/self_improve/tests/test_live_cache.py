from benchmarks.self_improve.live_cache import LiveResultCache, candidate_hash, run_config_hash


def test_candidate_hash_is_stable_across_dict_insertion_order():
    a = {"agents_md": "text a", "skills_tools_bash": "text b"}
    b = {"skills_tools_bash": "text b", "agents_md": "text a"}
    assert candidate_hash(a) == candidate_hash(b)


def test_candidate_hash_changes_with_a_single_character_difference():
    a = {"agents_md": "text a"}
    b = {"agents_md": "text A"}
    assert candidate_hash(a) != candidate_hash(b)


def test_run_config_hash_changes_when_any_field_changes():
    base = {"model": "m1", "max_attempts": 2, "thinking": None, "base_commit": "abc"}
    assert run_config_hash(base) != run_config_hash({**base, "model": "m2"})
    assert run_config_hash(base) != run_config_hash({**base, "max_attempts": 3})
    assert run_config_hash(base) != run_config_hash({**base, "base_commit": "def"})


def test_cache_miss_when_nothing_stored(tmp_path):
    cache = LiveResultCache(tmp_path)
    assert cache.get({"a": "1"}, {"model": "m"}, "python/wordy") is None


def test_cache_put_then_get_round_trips(tmp_path):
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    result = {"status": "pass_1", "score": 1.0}
    cache.put(candidate, run_config, "python/wordy", result)
    assert cache.get(candidate, run_config, "python/wordy") == result


def test_cache_key_changes_when_candidate_changes(tmp_path):
    cache = LiveResultCache(tmp_path)
    run_config = {"model": "m1"}
    cache.put({"agents_md": "text a"}, run_config, "python/wordy", {"status": "pass_1", "score": 1.0})
    assert cache.get({"agents_md": "text B"}, run_config, "python/wordy") is None


def test_cache_key_changes_when_run_config_changes(tmp_path):
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    cache.put(candidate, {"model": "m1"}, "python/wordy", {"status": "pass_1", "score": 1.0})
    assert cache.get(candidate, {"model": "m2"}, "python/wordy") is None


def test_cache_key_changes_when_exercise_id_changes(tmp_path):
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    cache.put(candidate, run_config, "python/wordy", {"status": "pass_1", "score": 1.0})
    assert cache.get(candidate, run_config, "python/bowling") is None


def test_environmental_results_are_never_cached(tmp_path):
    """Real safety property, not just a nice-to-have: caching an
    environmental failure (a provider timeout, a dead pi) would permanently
    pin a candidate at a false low score for the rest of the run."""
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    for status in ("error", "fail_timeout", "empty_response", "harness_error"):
        cache.put(candidate, run_config, "python/wordy", {"status": status, "score": 0.0})
        assert cache.get(candidate, run_config, "python/wordy") is None


def test_genuine_failure_is_still_cached(tmp_path):
    """"fail" (the model produced code, tests failed) is a genuine candidate
    outcome -- it must be cached, unlike an environmental failure."""
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    cache.put(candidate, run_config, "python/wordy", {"status": "fail", "score": 0.0})
    assert cache.get(candidate, run_config, "python/wordy") == {"status": "fail", "score": 0.0}


def test_corrupt_entry_is_treated_as_a_miss_not_an_exception(tmp_path):
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    path = cache._path_for(candidate_hash(candidate), run_config_hash(run_config), "python/wordy")
    path.parent.mkdir(parents=True)
    path.write_text("not valid json{{{")
    assert cache.get(candidate, run_config, "python/wordy") is None


def test_writes_are_atomic_no_stray_tmp_file_left_behind(tmp_path):
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    cache.put(candidate, run_config, "python/wordy", {"status": "pass_1", "score": 1.0})
    path = cache._path_for(candidate_hash(candidate), run_config_hash(run_config), "python/wordy")
    leftover_tmp = [p for p in path.parent.iterdir() if ".tmp-" in p.name]
    assert leftover_tmp == []


def test_exercise_id_with_slash_does_not_create_nested_directories(tmp_path):
    """Exercise ids look like "python/wordy" -- must not be interpreted as a
    path separator that escapes the intended candidate directory."""
    cache = LiveResultCache(tmp_path)
    candidate = {"agents_md": "text"}
    run_config = {"model": "m1"}
    cache.put(candidate, run_config, "python/wordy", {"status": "pass_1", "score": 1.0})
    cand_dir = cache._path_for(candidate_hash(candidate), run_config_hash(run_config), "python/wordy").parent
    assert list(cand_dir.iterdir()) == [cand_dir / "python__wordy.json"]
