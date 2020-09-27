import abc


class RecallChannel(metaclass=abc.ABCMeta):
    def __init__(self,rec_number):
        self.rec_number = rec_number

    @abc.abstractmethod
    def recall_df(self,df,params,number=30):
        pass

    @abc.abstractmethod
    def recall(self,params,number=30):
        pass

