def test_evaluation_and_inference_modules_import() -> None:
    from src.eval import evaluate
    from src.inference import inference

    assert callable(evaluate)
    assert callable(inference)
