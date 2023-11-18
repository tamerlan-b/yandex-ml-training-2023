import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.choice(data_length, size=data_length, replace=True))
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            bag_indices = self.indices_list[bag]
            data_bag, target_bag = np.take(data, bag_indices, axis=0), np.take(target, bag_indices, axis=0)
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        predictions = []
        for m in self.models_list:
            predictions.append(m.predict(data))
        predictions_np = np.array(predictions)
        return np.mean(predictions_np, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]

        # naive algorithm
        for i in range(len(list_of_predictions_lists)):
            for j in range(self.num_bags):
                bag_indices = self.indices_list[j]
                if not i in bag_indices:
                    p = self.models_list[j].predict([self.data[i]])
                    list_of_predictions_lists[i].append(p)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = []
        for i in range(len(self.list_of_predictions_lists)):
            if len(self.list_of_predictions_lists[i]) > 0:
                self.oob_predictions.append(np.mean(np.array(self.list_of_predictions_lists[i])))
            else:
                self.oob_predictions.append(None)

        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        oob_score = 0
        n = 0
        for i, oob_p in enumerate(self.oob_predictions):
            if not oob_p is None:
                oob_score += (self.target[i] - oob_p)**2
                n += 1
        return oob_score