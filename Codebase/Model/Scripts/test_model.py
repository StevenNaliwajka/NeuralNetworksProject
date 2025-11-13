def run_test_model(m: RunMetrics):
    ## Creates test data generator
    test_gen = m.datagen.flow_from_directory(
        ## Pull variables from train.
        m.output_path / "test",
        target_size=(128, 128),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )

    ## RUN + TIME
    t0 = time.time()
    ## Verbose 1 shows history.
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    t1 = time.time()

    ## compute times
    total_test_time = t1 - t0
    num_test_images = test_gen.samples
    steps_test = int(np.ceil(num_test_images / 32))
    avg_time_per_step = total_test_time / steps_test
    avg_time_per_image = total_test_time / num_test_images

    ## print times
    print(f"\nTotal inference time on test set: {total_test_time:.4f} s")
    print(f"Number of test images: {num_test_images}")
    print(f"Average time per step: {avg_time_per_step:.6f} s")
    print(f"Average time per image: {avg_time_per_image * 1000:.3f} ms")

    m.test_loss = float(test_loss)
    m.test_acc = float(test_acc)
    m.total_test_time = float(total_test_time)
    m.num_test_images = int(num_test_images)
    m.steps_test = int(steps_test)
    m.avg_time_per_step = float(avg_time_per_step)
    m.avg_time_per_image = float(avg_time_per_image)


run_test_model(m)
print("Finished")