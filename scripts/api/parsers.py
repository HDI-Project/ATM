def return_get_dataset_parser(api):
    comparison_args = [
        ('id', int), ('name', str), ('train_path', str), ('test_path', str),
        ('description', str), ('n_examples', int), ('k_classes', int),
        ('d_features', int), ('majority', float), ('size_kb', int)]
    operation_args = [
        ('n_examples_op', str), ('k_classes_op', str), ('d_features_op', str),
        ('majority_op', str), ('size_kb_op', str)]

    dataset_parser = api.parser()
    for col_tuple in comparison_args:
        dataset_parser.add_argument(col_tuple[0], type=col_tuple[1])
    for col_tuple in operation_args:
        dataset_parser.add_argument(
            col_tuple[0], type=col_tuple[1],
            help='comparison operator. i.e. =, >, >=')
    return dataset_parser


def return_set_dataset_parser(api):
    args = [
        ('name', str), ('description', str), ('train_path', str),
        ('test_path', str), ('class_column', str), ('n_examples', int),
        ('k_classes', int), ('d_features', int), ('majority', float)]

    dataset_parser = api.parser()
    for col in args:
        dataset_parser.add_argument(col[0], type=col[1])

    return dataset_parser