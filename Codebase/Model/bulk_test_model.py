## This can all be combined into one method.
def test_model(method_name):
    methods = {
        "1": inject_vgg_16,
        "2": inject_mobilenet_v2,
        "3": inject_lightweight_mobilenet_v2,
        "4": inject_vgg_5
    }
    model = methods[method_name]()
    target_size = model.input_shape[1:3]
    model.summary()
    print_params()
    train_model(m)
    print_total_time(m)
    print_loss(m)
    print_accuracy(m)
    run_test_model(m)
    print_test()
    save_params(model.name)


# VGG-16 already run above.
# test_model("1")
print("Finished")