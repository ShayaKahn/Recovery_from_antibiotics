class Subject:
    def __init__(self, **kwargs):
        self.base_array = kwargs.get('base_array')
        self.ant_array = kwargs.get('ant_array')
        self.int_array = kwargs.get('int_array')
        self.month_array = kwargs.get('month_array')
        self.base_days = kwargs.get('base_days')
        self.ant_days = kwargs.get('ant_days')
        self.int_days = kwargs.get('int_days')
        self.month_number = kwargs.get('month_number')

    @staticmethod
    def create_dict(array, days):
        if len(days) == 1:
            return {days[0]: array}
        else:
            columns = [array[:, i].tolist() for i in range(array.shape[1])]
            return dict(zip(days, columns))

    def create_base_dict(self):
        return self.create_dict(self.base_array, self.base_days)

    def create_ant_dict(self):
        return self.create_dict(self.ant_array, self.ant_days)

    def create_int_dict(self):
        return self.create_dict(self.int_array, self.int_days)

    def create_month_dict(self):
        return self.create_dict(self.month_array, self.month_number)