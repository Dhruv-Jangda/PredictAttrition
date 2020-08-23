from time import time
from numpy import float32, append, array
from pandas import DataFrame
from mxnet import gluon, init, autograd, ndarray, metric, sym, mod, io, mon


class DataManager:

    def get_batch_data(data: DataFrame, cols: dict, workers, batch_size,
                       style: str, is_test=False, shuffle=True):
        in_data = data[cols["continuous"] + cols["categorical"]["numeric"]
                       + cols["categorical"]["string"]].values.astype(float32)

        if style == "imperative":
            if is_test:
                data_set = gluon.data.ArrayDataset(in_data)
            else:
                target_data = data[cols["target"]].values.astype(float32)
                data_set = gluon.data.ArrayDataset(in_data, target_data)

            return gluon.data.DataLoader(
                dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                num_workers=workers, thread_pool=True)

        elif style == "symbolic":
            if is_test:
                data_iter = io.NDArrayIter(data=in_data, batch_size=batch_size,
                                           shuffle=shuffle, data_name="data")
            else:
                target_data = data[cols["target"]].values.astype(float32)
                data_iter = io.NDArrayIter(
                    data=in_data, label=target_data, batch_size=batch_size,
                    shuffle=shuffle, data_name="data", label_name="target",
                    last_batch_handle="roll_over")
            return data_iter

    def r_square(pred, label):
        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        return ndarray.sum(ndarray.square(pred - ndarray.mean(label))) /\
            ndarray.sum(ndarray.square(label - ndarray.mean(label)))


class ImperativeModel(DataManager):

    def __init__(self, loss_mae, smry_writer):
        self.writer = smry_writer
        self.loss_fn = loss_mae

    def prepare_model(self, num_features: int):
        self.model = gluon.nn.Sequential()
        self.model.add(
            gluon.nn.Dense(units=80, in_units=num_features, use_bias=True),
            gluon.nn.Dense(units=120, in_units=80, activation="relu"),
            gluon.nn.Dense(units=60, in_units=120, activation="relu",
                           use_bias=True),
            gluon.nn.Dense(units=20, in_units=60, activation="relu",
                           use_bias=True),
            gluon.nn.Dense(units=1, in_units=20, use_bias=True)
        )
        self.model.initialize(init=init.Xavier())

    def train_model(self, train_loader, epochs: int, optimizer: str,
                    learning_rate: float, batch_size: int, score_after: int):
        step: int = 0
        metric_rmse = metric.RMSE()
        trainer = gluon.Trainer(
            params=self.model.collect_params(),
            optimizer=optimizer,
            optimizer_params={"learning_rate": learning_rate}
        )
        for e in range(epochs):
            train_mae: float = 0.0
            train_rmse: float = 0.0
            r_square_val: float = 0.0
            iteration: int = 0

            for data, label in train_loader:
                start = time()
                iteration += 1
                step += 1

                # forward prop
                with autograd.record():
                    pred = self.model(data)
                    loss = self.loss_fn(pred.reshape([label.size, ]), label)

                # backward prop - performs d(Loss)/d(params)
                loss.backward()

                # update model parameters
                trainer.step(batch_size=batch_size)

                # training metrics
                metric_rmse.update([label], [pred])
                loss_mae = loss.mean().asscalar()
                loss_rmse = metric_rmse.get()
                train_mae += loss_mae
                train_rmse += loss_rmse[1]
                r_square_val += self.r_square(pred.reshape([label.size, ]),
                                              label).asscalar()

                # record metrics per batch to tensorboard
                self.writer.add_scalar(tag="Train_MAE",
                                       value=loss_mae, global_step=step)
                self.writer.add_scalar(tag="Train_RMSE",
                                       value=loss_rmse, global_step=step)

                if not bool(iteration % score_after):
                    print("Epoch % d; Iteration % d; MAE % .4f; "
                          "R-Square % .4f;  RMSE % .4f in %.3fs" %
                          (e, iteration, train_mae / iteration,
                           r_square_val / iteration, train_rmse / iteration,
                           time() - start))

            # record gradients per layer per epoch at tensorboard
            gradients: list = [
                param.grad() for param in self.model.collect_params().values()]
            param_names: list = list(self.model.collect_params().keys())
            for i, name in enumerate(param_names):
                self.writer.add_histogram(
                    tag=name, values=gradients[i], bins=1000, global_step=e)

    def validate_model(self, valid_loader):
        metric_rmse = metric.RMSE()
        valid_mae: float = 0.0
        valid_rmse: float = 0.0
        r_square_val: float = 0.0
        iteration: int = 0

        for data, label in valid_loader:
            start = time()
            iteration += 1

            with autograd.record():
                pred = self.model(data)
                loss = self.loss_fn(pred.reshape([label.size, ]), label)

            metric_rmse.update([label], [pred])
            loss_rmse = metric_rmse.get()
            loss_mae = loss.mean().asscalar()
            valid_mae += loss_mae
            valid_rmse += loss_rmse[1]
            r_square_val += self.r_square(pred.reshape(
                [label.size, ]), label).asscalar()

            # record metrics to tensorboard
            self.writer.add_scalar(
                tag="Validation_MAE", value=valid_mae, global_step=iteration)
            self.writer.add_scalar(
                tag="Validation_RMSE", value=valid_rmse,
                global_step=iteration)

            print("Iteration %d; MAE %.4f; R-Square %.4f ; RMSE %.4f"
                  " in %.3fs" % (iteration, valid_mae / iteration,
                                 r_square_val / iteration,
                                 valid_rmse / iteration,
                                 time()-start))

    def test_model(self, test_loader):
        iteration: int = 0
        for features in test_loader:
            results = self.model(features).reshape([len(features), ])
            results = array([pred.asscalar() for pred in results])
            if iteration == 0:
                target = results
            else:
                target = append(target, results)
            iteration += 1
        return target


class SymbolicModel(DataManager):

    def __init__(self, loss_mae, smry_writer):
        self.writer = smry_writer
        self.loss_fn = loss_mae

    def prepare_model(self):
        l1_dense = sym.FullyConnected(data=sym.Variable(
            "data"), num_hidden=80, no_bias=False, name="l1_dense")
        l2_dense = sym.FullyConnected(
            data=l1_dense, num_hidden=120, no_bias=True, name="l2_dense")
        l2_activation = sym.Activation(
            data=l2_dense, act_type="relu", name="l2_activation")
        l3_dense = sym.FullyConnected(
            data=l2_activation, num_hidden=60, no_bias=True, name="l3_dense")
        l3_activation = sym.Activation(
            data=l3_dense, act_type="relu", name="l3_activation")
        l4_dense = sym.FullyConnected(
            data=l3_activation, num_hidden=20, no_bias=True, name="l4_dense")
        l4_activation = sym.Activation(
            data=l4_dense, act_type="relu", name="l4_activation")
        self.l5_dense = sym.FullyConnected(
            data=l4_activation, num_hidden=1, no_bias=False, name="l5_dense")
        output = sym.MAERegressionOutput(
            data=self.l5_dense, label=sym.Variable("target"))
        self.train_module = mod.Module(
            symbol=output, data_names=["data"], label_names=["target"])

    def train_model(self, train_iter, batch_size: int, epochs: int,
                    num_features, optimizer: str, learning_rate: float,
                    momentum: float, score_after: float, eval_metric: str):
        """
        Parameters
        ----------
        eval_metric - "accuracy", "ce" (CrossEntropy), "f1", "mae", "mse",
                      "rmse", "top_k_accuracy".
        """
        # Set Monitor
        monitor = mon.Monitor(interval=score_after, pattern=".*",
                              stat_func=self.loss_fn)
        self.train_module.install_monitor(monitor)

        # Create Metric
        eval_metric_fn = metric.create(eval_metric)
        self.train_module.bind(
            data_shapes=io.DataDesc(name="data",
                                    shape=(batch_size, num_features)),
            label_shapes=io.DataDesc(name="target",
                                     shape=(batch_size, 1)),
            for_training=True)
        self.train_module.init_optimizer(
            optimizer=optimizer,
            optimizer_params={"learning_rate": learning_rate,
                              "momentum": momentum}
        )
        self.train_module.init_params()

        for epoch in range(epochs):
            eval_metric_fn.reset()
            end_of_batch = False
            tic = time()
            nbatch = 0
            while not end_of_batch:
                try:
                    train_batch = train_iter.next()
                    monitor.tic()
                    self.train_module.forward_backward(train_batch)
                    self.train_module.update()
                    self.train_module.update_metric(
                        eval_metric_fn, train_batch.label)

                    for name, value in eval_metric_fn.get():
                        self.writer.add_scalar(
                            tag="Train " + name, value=value,
                            global_step=(epoch+1)*nbatch)
                        print("Epoch[%d] Batch[%d] Train-%s=%.3f" %
                              (epoch, nbatch, name, value))
                except Exception:
                    end_of_batch = True
                nbatch += 1

            print("Epoch[%d] completed! Time cost=%.3f s" %
                  (epoch, (time()-tic)))

            for grad in self.train_module.get_input_grads():
                self.writer.add_histogram(
                    values=grad, bins=1000, global_step=epoch)

        # # Direct
        # self.train_module.fit(train_data=train_iter, eval_data=valid_iter,
        #                       num_epoch=epochs, eval_metric=metric,
        #                       validation_metric=metric, monitor=monitor)

    def validate_model(self, valid_iter, eval_metric):
        """
        Parameters
        ----------
        eval_metric - "accuracy", "ce" (CrossEntropy), "f1", "mae", "mse",
                      "rmse", "top_k_accuracy".
        """
        # res = self.train_module.score(eval_data, validation_metric)

        eval_metric_fn = metric.create(eval_metric)
        end_of_batch = False
        nbatch = 0
        while not end_of_batch:
            try:
                valid_batch = valid_iter.next()
                self.train_module.forward(valid_batch)
                self.train_module.update_metric(
                    eval_metric_fn, valid_batch.label)
                for name, value in eval_metric_fn.get():
                    self.writer.add_scalar(
                        tag="Validation " + name, value=value,
                        global_step=nbatch)
                    print("Batch[%d] Validation-%s=%.3f" %
                          (nbatch, name, value))
            except Exception:
                end_of_batch = True
            nbatch += 1

    def test_model(self, test_iter, batch_size: int, num_features):
        test_module = mod.Module(symbol=self.l5_dense, data_names=[
            "data"], label_names=None)
        test_module.bind(
            data_shapes=io.DataDesc(name="data",
                                    shape=(batch_size, num_features)),
            for_training=False, shared_module=self.train_module)
        pred = test_module.predict(eval_data=test_iter)
        return pred.asnumpy()


# Wrapper class around both ImperativeModel and SymbolicModel
class ModelManager():

    def __init__(self, model_style: str, loss_fn):
        self.style = model_style
        self.loss = loss_fn

    def get_batch_data(self, data: DataFrame, cols: dict, num_workers, label,
                       batch_size, style: str, is_test=False, shuffle=True):
        print("\nPreparing {} data...".format(label))
        data_mgr = DataManager()
        return data_mgr.get_batch_data(
            data=data, cols=cols, workers=num_workers, batch_size=batch_size,
            style=style, is_test=False, shuffle=True)
        print("Data prepared!")

    def prepare_model(self, writer):
        print("Preparing model...")
        if self.style == "imperative":
            self.imp_model = ImperativeModel(
                loss_mae=self.loss, smry_writer=writer)
            self.imp_model.prepare_model()
            print("Model prepared. Structure -\n{}".format(
                self.imp_model.model.collect_params))

        elif self.style == "symbolic":
            self.sym_model = SymbolicModel(
                loss_mae=self.loss, smry_writer=writer)
            self.sym_model.prepare_model()
            print("Model prepared. Structure -\n{}".format(
                self.sym_model.output.collect_params))

    def train_model(self, train_data, score_after, epochs,
                    lrn_rate: float, batch_size, optimizer: str,
                    num_features, momentum: float, eval_metric: str):
        print("\nTraining model...")
        if self.style == "imperative":
            self.imp_model.train_model(
                train_loader=train_data, epochs=epochs, learning_rate=lrn_rate,
                batch_size=batch_size, score_after=score_after)
        elif self.style == "symbolic":
            self.sym_model.train_model(
                train_iter=train_data, batch_size=batch_size, epochs=epochs,
                num_features=num_features, optimizer=optimizer,
                learning_rate=lrn_rate, momentum=momentum, eval_metric=metric)
        print("Model trained!")

    def validate_model(self, valid_data, metric: str):
        print("Validating trained model...")
        if self.style == "imperative":
            self.imp_model.validate_model(valid_loader=valid_data)
        else:
            self.sym_model.validate_model(
                valid_iter=valid_data, eval_metric=metric)
        print("Model validated!")

    def test_model(self, test_data, batch_size, features: int):
        print("Testing model...")
        if self.style == "imperative":
            return self.imp_model.test_model(test_loader=test_data)
        elif self.style == "symbolic":
            return self.sym_model.test_model(
                test_iter=test_data, batch_size=batch_size,
                num_features=features)
        print("Model Tested")

    def save_results(self, cols, data, path):
        test_results: DataFrame = DataFrame(
            columns=[cols["id"], cols["target"]])
        test_results[cols["id"]] = data["id"]
        test_results[cols["target"]] = data["target"]
        test_results.to_csv(path, index=False)
        print("Results available at {}".format(path))

    def save_model(self, path: dict):
        if self.style == "imperative":
            self.imp_model.model.save_parameters(path["params"])
            with open(path["arch"], mode="w+") as txt_file:
                txt_file.write(str(self.imp_model.model.collect_params()))
        elif self.style == "symbolic":
            self.sym_model.train_module.save_params(path["params"])
