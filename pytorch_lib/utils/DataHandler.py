import torch


class DataHandler:

    @staticmethod
    def split(dataset, test_frac):

        test_size = int(test_frac * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset

    @staticmethod
    def even_split(dataset, test_frac):
        # @todo
        pass

    @staticmethod
    def predict_data_loader(model, data_loader, device='cpu', return_true=False, model_args={}):
        with torch.no_grad():
            model.eval()
            model.to(device)

            y_pred = []
            y_true = []

            for data, y in data_loader:
                data = data.to(device)
                y = y.to(device)

                yp = model.predict(data)
                y_pred += [yp]
                y_true += [y]

            if return_true:
                return torch.cat(y_pred), torch.cat(y_true)
            return torch.cat(y_pred)
