def return_get_dataset_parser(api):
    operation_args = [
        ('n_examples_op', str), ('k_classes_op', str), ('d_features_op', str),
        ('majority_op', str), ('size_kb_op', str)]

    dataset_parser = return_set_dataset_parser(api)
    dataset_parser.add_argument('id', int, help='exact matches only')

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


def return_put_dataset_parser(api):
    dataset_parser = return_set_dataset_parser(api)
    dataset_parser.add_argument('id', int, help='exact matches only')

    replacement_args = [
        ('new_name', str), ('new_description', str), ('new_train_path', str),
        ('new_test_path', str), ('new_class_column', str), ('new_n_examples', int),
        ('new_k_classes', int), ('new_d_features', int), ('new_majority', float)]

    for col in replacement_args:
        dataset_parser.add_argument(col[0], type=col[1])

    return dataset_parser
