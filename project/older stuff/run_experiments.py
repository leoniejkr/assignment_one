from file_handling import load_input
from runner import ExperimentRunner
from analysis_stuff.analyser import ExperimentAnalyzer
from analysis_stuff.vizualiser import ExperimentVisualizer
from analysis_stuff.ana_plot import ExperimentVisualizerAnalyzer
from analysis_stuff.comprehensive_analysis import ComprehensiveAnalyzer

if __name__ == "__main__":
    data = load_input("project/data/input/busy_day.in")
    runner = ExperimentRunner(data)
    runner.run_all(num_runs=1)

    # analyzer = ExperimentAnalyzer("project/results")
    # analyzer.full_analysis()

    # visualizer = ExperimentVisualizer(runner.results, output_dir="project/results/plots")
    # visualizer.plot_score_distribution()
    # visualizer.plot_runtime_vs_score()
    # visualizer.plot_convergence_curves()
    # visualizer.plot_average_performance()
    # visualizer.plot_efficiency()
    # visualizer.plot_parameter_sensitivity()
    # visualizer.save_algorithm_ranking()
    # visualizer.statistical_significance_tests()

    # analyzer_visualizer = ExperimentVisualizerAnalyzer(runner.results, "project/results/plots")
    # analyzer_visualizer.full_analysis()

    comprehensive_analyzer = ComprehensiveAnalyzer("project/results")
    comprehensive_analyzer.run_full_analysis()




    
    
