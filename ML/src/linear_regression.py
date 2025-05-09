"""Module which implements Linear Regression Model."""

import random 
import enum
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any, Callable, ClassVar
import functools
import inspect

import numpy as np
import pydantic

from Internals.logger import logger

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionLoss(enum.Enum):
    """Enumerates avaliable  regression loss functions."""

    MSE = 'mse'
    MAE = 'mae'
    SINMSE = 'sinmse'
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionLossFunctionI(ABC):
    """Defines interface of regression loss function."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates difference between true target label and predicted label with defined loss function."""

class MSE(RegressionLossFunctionI):
    """Implements mean squared error loss function."""

    @override
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates mean  squeared error difference between true target label and predicted label.Formula: f(y_true, y_pred) = np.mean((y_true -  y_pred)^2).

        Args:
            y_true - True target label.
            y_pred - Label predicted with ML model.

        Returns:
            int | float: mean  squeared error difference between true target label and predicted label.
        """
        return np.mean((y_true -  y_pred)**2)
    
    def __repr__(self)  -> str:
        return f'MSE()'
    

class MAE(RegressionLossFunctionI):
    """Implements mean absolute error loss function."""

    @override
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates mean  absolute error difference between true target label and predicted label.Formula: f(y_true, y_pred) = np.mean(np.abs(y_true -  y_pred)).

        Args:
            y_true - True target label.
            y_pred - Label predicted with ML model.

        Returns:
            int | float: mean absolute error difference between true target label and predicted label.
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def __repr__(self) -> str:
        return f'MAE'
    
class SinMSE(RegressionLossFunctionI):
    """Implements mean squared sinus of mean squeared error loss function."""

    @override
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> int | float:
        """Calculates  mean squared sinus of mean squeared error difference between true target label and predicted label.Formula: f(y_true, y_pred) = np.mean(sin((y_true -  y_pred)^2)^2).

        Args:
            y_true - True target label.
            y_pred - Label predicted with ML model.

        Returns:
            int | float: mean squared sinus of mean squeared error difference between true target label and predicted label
        """
        return np.mean(np.sin((y_true - y_pred)**2)**2)
    
    def __repr__(self) -> str:
        return 'SinMSE()'
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegressionGradientI(ABC):
    """Defines interface of loss function gradient calculation."""

    @abstractmethod
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        ...

class MSEGradient(RegressionGradientI):
    """Calculate gradient of  MSE loss function."""

    @override
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        """Caltulatespartial derivatives of  MSE loss function.

        Args:
            X: Input training set.
            y: Tarter features.
            prediction: Result of  model  forward pass with  input  X
            bias: Indicate if include  bias parameter in forward pass or not.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with partial derivatives of  MSE loss function.
        """
        if bias:
            return ((-2*X.T) @ (y - prediction) / X.shape[0], -2*(y - prediction)/ X.shape[0])
        else:
            return ((-2*X.T) @ (y - prediction)/ X.shape[0], np.zeros(X.shape[0]))

    def __repr__(self) -> str:
        return f'MSEGradient()'
    
class MAEGradient(RegressionGradientI):
    """Calculate gradient of  MAE loss function."""

    @override
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        """Caltulatespartial derivatives of  MAE loss function.

        Args:
            X: Input training set.
            y: Tarter features.
            prediction: Result of  model  forward pass with  input  X
            bias: Indicate if include  bias parameter in forward pass or not.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with partial derivatives of  MAE loss function.
        """
        sign = np.sign(y - prediction)
        return (X.T @ (-sign),  -sign) if bias else (X.T @ (-sign), np.array([0.0]))
    
    def __repr__(self) -> str:
        return f'MAEGradient()'
    
class SinMSEGradient(RegressionGradientI):
    """Calculate gradient of  SinMSE loss function."""

    @override
    def __call__(self, X: np.ndarray, y: np.ndarray, prediction: np.ndarray, bias: bool) -> tuple[np.ndarray, np.ndarray]:
        """Caltulates partial derivatives of  SinMSE loss function.

        Args:
            X: Input training set.
            y: Tarter features.
            prediction: Result of  model  forward pass with  input  X
            bias: Indicate if include  bias parameter in forward pass or not.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with partial derivatives of  SinMSE loss function.
        """
        mse_loss = (y - prediction)**2
        if bias:
            return  (X.T @ (-4*np.sin(mse_loss)*np.cos(mse_loss)) / X.shape[0], (-4*np.sin(mse_loss)*np.cos(mse_loss)) / X.shape[0])
        else:
            return   (X.T @ (-4*np.sin(mse_loss)*np.cos(mse_loss)) / X.shape[0], np.zeros(X.shape[0]))
        
    def __repr__(self) -> str:
        return f'SinMSEGradient()'
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class EarlyStopping(enum.Enum):
    """Enumerates avaliable early stopping mechanisms."""

    GRADIENT_NORM = 'Gradient norm'
    PARAMETERS_CHANGE = 'Parameters change'
    FUNCTION_CHANGE =  'Function change'

class EarlyStoppingI(ABC):
    """Defines interface for early stopping of  gradient  descent."""

    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        """Perform required check do define either to continue gradient descent or stop early.

        Args:
            params_before_step: model parameters before gradient step.
            grad: partial derivatives with respect to model parameters withing defined loss  function.
            params_after_step: model parameters after  gradient step.
            prediction_before: model prediction before gradient step.
            predictoin_after: model prediction after gradient step.

        Returns:
            bool: True if to stop gradient descent early, False otherwise.
        """

class GradientNormStopping(EarlyStoppingI):
    """Stops gradient descent early when gradient norm is smaller then defined treshold."""

    @override
    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray, np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        for partial_derivative in grad:
            gradient_norm = np.abs(np.linalg.norm(partial_derivative))
            if gradient_norm != 0 and gradient_norm < treshold:
                return True
        return False

    def __repr__(self) -> str:
        return f'GradientNormStopping()'
    
class ParametersChangeStopping(EarlyStoppingI):
    """Stops gradient descent early when parameters change is smaller then defined treshold."""

    @override
    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray, np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        for param_before_step, param_after_step in zip(params_before_step, params_after_step):
            param_change = np.mean(np.abs(param_after_step - param_before_step))
            if param_change != 0 and param_change < treshold:
                return True
        return False
    
    def __repr__(self) -> str:
        return f'ParametersChangeStopping()'
    
class FunctionChangeStopping(EarlyStoppingI):
    """Stops gradient descent when model output change is smaller then defined treshold."""

    @override
    def __call__(self, 
                 params_before_step: np.ndarray,
                 grad: tuple[np.ndarray, np.ndarray],
                 params_after_step: np.ndarray,
                 prediction_before: np.ndarray,
                 prediction_after: np.ndarray,
                 treshold: int | float
                 ) -> bool:
        function_change = np.mean(np.abs(prediction_after - prediction_before))
        return True if function_change != 0 and  function_change < treshold else False
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ParamsInitialization(enum.Enum):
    """Enumaration of avaliable params initialization functions."""
    RANDOM_UNIFORM =  'Random uniform'
    CONSTANT  = 'constant'

# Type alias for params initialization functions
PARAMS_INITIALIZER = Callable[[tuple[int, int]], tuple[np.array, np.array]]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ValidateType:
    """Decorator to validate types of arguments provided to a function.
    
    Args:
        expected_type: (tuple[str, Any] | list[tuple[str, Any]]):  expected type for provided argument.
    """
    
    def __init__(self, expected_type: tuple[str, Any] | list[tuple[str, Any]]):
        self.expected_type = expected_type
    
    @staticmethod
    def _validate_arg(arg_name: Any, expected_type: Any, kwargs: dict) -> None:
        arg = kwargs[arg_name]
        if not isinstance(arg, expected_type) and not (inspect.isclass(arg) and 
                                                       issubclass(arg, expected_type)):
            raise TypeError(f'Invalid input type for {arg_name}: expected: {expected_type}')    
      
    @staticmethod      
    def _check_missing_arguments(expected_type, provided_type) -> None:
        # wrap expected type into list to avoid separate logic for not list case
        if not isinstance(expected_type, list):
            expected_type = [expected_type]
            
        for arg, _ in expected_type:
            if not arg in provided_type:
                raise KeyError(f'Required keyword argument {arg} is not found.')
                    
    def __call__(self, func: Callable) -> Callable:
        """Decorator call method that applies validation logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._check_missing_arguments(self.expected_type, kwargs)
            
            if isinstance(self.expected_type, list):
                for arg, expected_type in self.expected_type:
                    self._validate_arg(arg, expected_type, kwargs)   
            else:
                self._validate_arg(kwargs[self.expected_type[0]], self.expected_type[1])   
            return func(*args, **kwargs)
        return wrapper
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Model(ABC):
    """Base class for machine learning models."""
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to set X."""
        ...
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass of a model."""
        ...
        
class LinearRegression(Model, pydantic.BaseModel):
    """Implements linear regression machine learning model. Forward pass: f(x) = x * w  + b.

    Args:
        learning_rate: converage step.
        n_iters: Number of backprogation iterations.
        bias: Indicate if include  bias parameter in forward pass or not.
        loss:  Define which loss function will be minimized.
        early_stopping: Define which early stopping mechanism to use.
        early_stopping_treshold: Defines treshold for early stopping.
        params_initialization: Defines how params will be initialized.
    """
    model_config =  pydantic.ConfigDict(arbitrary_types_allowed=True)

    _losses: ClassVar[dict[RegressionLossFunctionI, tuple[RegressionLossFunctionI, RegressionGradientI]]] =  {
        RegressionLoss.MSE: (MSE(), MSEGradient()),
        RegressionLoss.MAE: (MAE(), MAEGradient())
        }
    
    _early_stoppings : ClassVar[dict[str, EarlyStoppingI]] = {
        EarlyStopping.GRADIENT_NORM:  GradientNormStopping(),
        EarlyStopping.PARAMETERS_CHANGE: ParametersChangeStopping(),
        EarlyStopping.FUNCTION_CHANGE: FunctionChangeStopping()
        }
    
    _params_initializations: ClassVar[dict[str, PARAMS_INITIALIZER]] = {
         ParamsInitialization.RANDOM_UNIFORM: lambda input_dims: (np.random.uniform(-1, 1, input_dims[-1]), np.zeros(input_dims[0])),
         ParamsInitialization.CONSTANT: lambda input_dims: (np.full(shape=input_dims[-1], fill_value=0.0), np.full(shape=input_dims[0], fill_value=0.0))
    }

    learning_rate: int | float = pydantic.Field(default=0.001)
    n_iters: int =  pydantic.Field(default=10)
    bias: bool = pydantic.Field(default=True)
    loss: RegressionLoss = pydantic.Field(default=RegressionLoss.MSE)
    early_stopping: EarlyStopping = pydantic.Field(default=EarlyStopping.GRADIENT_NORM)
    early_stopping_treshold: float = pydantic.Field(default=0.00001)
    params_initialization: ParamsInitialization = pydantic.Field(default=ParamsInitialization.RANDOM_UNIFORM)
    w: np.ndarray = pydantic.Field(default=None)
    b: np.ndarray = pydantic.Field(default=None)

    def model_post_init(self, context):
        self._loss_fn, self._grad = self._losses[self.loss]
        self._early_stopping = self._early_stoppings[self.early_stopping]
        self._params_initialization = self._params_initializations[self.params_initialization]

    def __repr__(self):
        return f'LinearRegression(learning_rate={self.learning_rate}, n_iters={self.n_iters}, bias={self.bias}, loss={self.loss})'
    
    @classmethod
    @ValidateType(expected_type=[('loss_function_name',  RegressionLoss),  ('loss_function', RegressionLossFunctionI), ('grad',  RegressionGradientI)])
    def _register_loss(cls, 
                       loss_function_name: RegressionLoss,
                       loss_function: RegressionLossFunctionI, 
                       grad: RegressionGradientI
                       ) -> None:
        """Register new loss which can be optimized to fine optimizal weight of the model.

        Args:
            loss_function_name: Key on  which  new  loss function will be stored.
            loss_function: Object which will calculate loss with defined formula.
            grad: Object which calculates gradies of provided loss function  with respect to model  parameters.
        """
        cls._losses[loss_function_name]  =  (loss_function(), grad())

    @classmethod
    @ValidateType(expected_type=[('early_stopping_name',  EarlyStopping),  ('early_stopping', EarlyStoppingI)])
    def _register_early_stopping(cls, early_stopping_name: EarlyStopping, early_stopping: EarlyStoppingI) -> None:
        """Register new  early stopping.

        Args:
            early_stopping_name: Key on which new early stopping will be stored.
            early_stopping: Object that performs early stopping.

        """
        cls._early_stoppings[early_stopping_name] = early_stopping()

    @classmethod
    def _register_params_initialization(cls, params_initialization_name: ParamsInitialization, params_initialization: PARAMS_INITIALIZER):
        """Register new params initialization function.

        Args:
            params_initialization_name: Key on which  new params initialization  function will be stored.
            params_initialization: Function thath initialize model params.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with initialized weights and bias.
        """
        cls._params_initializations[params_initialization_name] = params_initialization
    
    def _step(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Perform one step of gradient descent with respect  to provided loss function.

        Args:
            X: Set to fit model to.
            y: Targets.
        """
        params_before_step = [self.w.copy(),  self.b.copy()]
        prediction_before = self.predict(X=X)
        dw, db = self._grad(X, y, self.predict(X=X), self.bias)
        self.w -= self.learning_rate * dw
        self.b -= self.b - self.learning_rate * db
        grad = [dw, db]
        params_after_step = [self.w, self.b]
        prediction_after = self.predict(X=X)
        return self._early_stopping(params_before_step=params_before_step, 
                                    grad=grad,
                                    params_after_step=params_after_step,
                                    prediction_before=prediction_before,
                                    prediction_after=prediction_after,
                                    treshold=self.early_stopping_treshold
                                    )

    @override
    @ValidateType(expected_type=[('X', np.ndarray), ('y', np.ndarray), ('verbose', bool)])
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """Iterativaly perfrom gradient descent steps with respect to specified  loss  function.

        Args:
            X: Set to fit model to.
            y: Targets.
            verbose: Show loss function value on each  iteration, if True.
        """
        self.w, self.b = self._params_initialization(X.shape)
        for iteration in range(self.n_iters):
            if verbose: logger.info(f'Loss {iteration}: {self._loss_fn(y, self.predict(X=X))}')
            if self._step(X, y): break

    @override  
    @ValidateType(expected_type=[('X', np.ndarray)])
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass of linear regression.

        Args:
            X: input vector for forward pass.
        
        Returns:
            np.ndarray: Result of forward pass with input matrix.
        """
        return X @ self.w + self.b[0] if self.bias else X @ self.w
        
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Register custom loss function to LinearRegression
    LinearRegression._register_loss(loss_function_name=RegressionLoss.SINMSE,
                                    loss_function=SinMSE,
                                    grad=SinMSEGradient)
    
    # Initialize model with specific loss function
    model = LinearRegression(learning_rate=0.001,
                             n_iters=1000,
                             loss=RegressionLoss.SINMSE,
                             early_stopping=EarlyStopping.GRADIENT_NORM,
                             early_stopping_treshold=0.0000001,
                             params_initialization=ParamsInitialization.RANDOM_UNIFORM
                             )
    # Initialize  training  set
    X = np.array([
        [1, 1], 
        [2, 2], 
        [3, 3], 
        [4, 4]
        ])
    y = np.array([2, 4, 6, 8])

    # Fit the model and make prediction.
    model.fit(X=X, y=y, verbose=True)
    print(f'Predicting set:', model.predict(X=X))
    print(f'Predicting sample:', model.predict(X=np.array([5,5])))