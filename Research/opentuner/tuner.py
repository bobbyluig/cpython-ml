import opentuner
from opentuner import ConfigurationManipulator
from opentuner import LogIntegerParameter
from opentuner import EnumParameter
from opentuner import MeasurementInterface
from opentuner import Result

FILE = '../programs/transaction/transaction2.py'
INTERPRETER = '../../../python3/bin/python3.7'
MODULO_THRESHOLD = 500
TIME_LIMIT = 3.0


class GCTuner(MeasurementInterface):
    def __init__(self, *args, **kwargs):
        super(GCTuner, self).__init__(*args, **kwargs)
        self.parallel_compile = True

        base_cmd = '{} {}'.format(INTERPRETER, FILE)
        run_result = self.call_program(base_cmd)
        assert run_result['returncode'] == 0

        values = [value.strip() for value in run_result['stderr'].split()]
        self.default_memory = int(values[0])
        self.default_objects = int(values[1])

        param_cmd = 'RESEARCH_MODULO1={} RESEARCH_VERBOSE=1 {} {}'.format(MODULO_THRESHOLD, INTERPRETER, FILE)
        run_result = self.call_program(param_cmd)
        assert run_result['returncode'] == 0

        values = [value.strip() for value in run_result['stderr'].split()]
        self.instructions = [int(pair.split(':')[0]) for pair in values[2].split(',')[:-1]]
        print('Located {} possible collection points.'.format(len(self.instructions)))

    def manipulator(self):
        m = ConfigurationManipulator()
        m.add_parameter(EnumParameter('instruction1', self.instructions))
        m.add_parameter(LogIntegerParameter('modulo1', 5, 1000000))
        m.add_parameter(EnumParameter('generation1', [0, 1, 2]))
        m.add_parameter(EnumParameter('instruction2', self.instructions))
        m.add_parameter(LogIntegerParameter('modulo2', 5, 1000000))
        m.add_parameter(EnumParameter('generation2', [0, 1, 2]))
        return m

    def compile(self, cfg, id):
        run_cmd = 'RESEARCH_INSTRUCTION1={} RESEARCH_MODULO1={} RESEARCH_GENERATION1={} ' \
                  'RESEARCH_INSTRUCTION2={} RESEARCH_MODULO2={} RESEARCH_GENERATION2={} ' \
                  '{} {}'.format(cfg['instruction1'], cfg['modulo1'], cfg['generation1'], cfg['instruction2'],
                                 cfg['modulo2'], cfg['generation2'], INTERPRETER, FILE)
        try:
            run_result = self.call_program(run_cmd, limit=TIME_LIMIT)
            assert run_result['returncode'] == 0
        except:
            return Result(time=float('inf'))

        values = [int(value.strip()) for value in run_result['stderr'].split()]
        memory = values[0]
        objects = values[1]

        if memory >= self.default_memory:
            score = 2000000000000000000 + memory
        elif objects >= self.default_objects:
            score = 1000000000000000000 + objects
        else:
            score = objects

        return Result(time=score)

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        return compile_result

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        cfg = configuration.data
        print('Optimal values written to optimal.json: {}'.format(cfg))
        self.manipulator().save_to_file(cfg, 'optimal.json')

        run_cmd = 'RESEARCH_INSTRUCTION1={} RESEARCH_MODULO1={} RESEARCH_GENERATION1={} ' \
                  'RESEARCH_INSTRUCTION2={} RESEARCH_MODULO2={} RESEARCH_GENERATION2={} ' \
                  '{} {}'.format(cfg['instruction1'], cfg['modulo1'], cfg['generation1'], cfg['instruction2'],
                                 cfg['modulo2'], cfg['generation2'], INTERPRETER, FILE)
        run_result = self.call_program(run_cmd)
        assert run_result['returncode'] == 0

        values = [value.strip() for value in run_result['stderr'].split()]
        optimal_memory = int(values[0])
        optimal_objects = int(values[1])

        print('Default strategy: {} {}'.format(self.default_memory, self.default_objects))
        print('Optimal strategy: {} {}'.format(optimal_memory, optimal_objects))

        if optimal_memory < self.default_memory and optimal_objects < self.default_objects:
            print('Pareto dominant solution found.')
        else:
            print('Failed to find pareto dominant solution.')


if __name__ == '__main__':
    arg_parser = opentuner.default_argparser()
    GCTuner.main(arg_parser.parse_args())
