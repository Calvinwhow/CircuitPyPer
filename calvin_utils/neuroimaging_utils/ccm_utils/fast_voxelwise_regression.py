import os
import glob 

indep_var_list = ['Gait', 'HeelToShinTestLeft', 'HeelToShinTestRight', 'FingerToNoseTestLeft', 'FingerToNoseTestRight', 'LimbAtaxia', 'Speech', 'Oculomotor', 'TotalBarsScore', 'SematicFluencyRawS', 'SematicFluencyFailS', 'PhonemicFluencyRawS', 'PhonemicFluencyFailS', 'CategorySwitchRawS', 'CategorySwitchFailS', 'VerbalRegSum', 'DigitSpanForwardRawS', 'DigitSpanForwardFailS', 'DigitSpanBackwardRawS', 'DigitSpanBackwardFailS', 'CubeDrawRawS', 'CubeDrawFailS', 'VerbalRecallRawS', 'VerbalRecallFailS', 'SimiliarityPair1RawS', 'SimiliarityPair2RawS', 'SimiliarityPair3RawS', 'SimiliarityPair4RawS', 'SimiliarityRawS', 'SimiliarityFailS', 'GoNoGoRawS', 'GoNoGoFailS', 'AFFECTAssessments_Angryoraggress', 'AFFECTAssessments_Difficultywith', 'AFFECTAssessments_Emotionallylab', 'AFFECTAssessments_Expressesillog', 'AFFECTAssessments_Lacksempathyis', 'AFFECTAssessments_Showseasysenso', 'AffectRawS', 'AffectFailS', 'TotalCCASRawScore', 'TotalCCASFailScore', 'MemoryTotal', 'ExecutiveTotal', 'LanguageTotal', 'VisualTotal', 'AffectTotal', 'delta_days_ccas_y', 'Sec1ADifficultFocus', 'Sec1AEasilyDistracted', 'Sec1AOntheGo', 'Sec1AFeelsCompelled', 'Sec1AFeelsDriven', 'Sec1BWorries', 'Sec1BRepeats', 'Sec1BMentallyStuck', 'Sec1BCauseDistress', 'Sec2AActHastily', 'Sec2ARapidChanges', 'Sec2ACryingLaughing', 'Sec2AOverAnxious', 'Sec2BLackOfPleasure', 'Sec2BNegativeAttitude', 'Sec2BUneasyWithLife', 'Sec2BSadDepressed', 'Sec3ARepetitiveMovements', 'Sec3ASensoryExp', 'Sec3BSensitive', 'Sec3BOverwhelmed', 'Sec4ACommunicates', 'Sec4AConcerns', 'Sec4ASeesHearsThings', 'Sec4BTroubleUnderstand', 'Sec4BDistant', 'Sec4BIndifferent', 'Sec5AAngry', 'Sec5AUpset', 'Sec5AIntolerant', 'Sec5AArgumentative', 'Sec5Bimmature', 'Sec5BUnaware', 'Sec5BManner', 'Sec5BTrusting', 'ScoreCol1A', 'ScoreCol1B', 'ScoreCol2A', 'ScoreCol2B', 'ScoreCol3A', 'ScoreCol3B', 'ScoreCol4A', 'ScoreCol4B', 'ScoreCol5A', 'ScoreCol5B', 'TotalSection1Score', 'TotalSection2Score', 'TotalSection3Score', 'TotalSection4Score', 'TotalSection5Score', 'CNRSTotColAScore', 'CNRSTotColBScore', 'CNRSTotScore']
for indep_var in indep_var_list:
    # Specify where you want to save your results to
    out_dir = '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/results/schmahmann/connectivity_r_maps'

    # Specify the path to your CSV file containing NIFTI paths
    input_csv_path = '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/resources/datasets/Schahmann_SCA_Atrophy/metadata/BARS-CCAS-CNRS_nearest_higher-is-better.csv'
    sheet = None

    from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
    # Instantiate the PalmPrepararation class
    cal_palm = CalvinStatsmodelsPalm(input_csv_path=input_csv_path, output_dir=out_dir, sheet=sheet)
    # Call the process_nifti_paths method
    data_df = cal_palm.read_data()
    drop_list = ['Nifti_File_Path', indep_var]
    data_df = cal_palm.drop_nans_from_columns(columns_to_drop_from=drop_list)

    # Variables to Drop by Row Values
    # column = 'Dataset'  # The column you'd like to evaluate
    # condition = 'equal'  # The condition to check ('equal', 'above', 'below', 'not')
    # value = 'PD STN DBS' # The value to drop if found
    # data_df, other_df = cal_palm.drop_rows_based_on_value(column, condition, value)

    # Remove anything you don't want to standardize
    cols_not_to_standardize = ['Nifti_File_Path', 'subject']
    group_col = 'Dataset'

    # data_df = cal_palm.standardize_columns(cols_not_to_standardize, group_col=group_col)


    # Begin Regression
    # Set this to the single variable you want to analyze
    out_dir_indep_var = os.path.join(out_dir, indep_var)
    formula = f"Nifti_File_Path ~ {indep_var}"

    # Define the design matrix
    outcome_matrix, design_matrix = cal_palm.define_design_matrix(formula, data_df=data_df, voxelwise_variable_list=['Nifti_File_Path'], coerce_str=False)
    contrast_matrix = cal_palm.generate_basic_contrast_matrix(design_matrix)
    contrast_matrix_df = cal_palm.finalize_contrast_matrix(design_matrix=design_matrix, contrast_matrix=contrast_matrix) 

    mask_path = '/Users/cu135/Software_Local/calvin_utils_project/calvin_utils_project/resources/MNI152_T1_2mm_brain_mask.nii'
    data_transform_method='standardize'
    if glob.glob(os.path.join(out_dir_indep_var, '**', '*.nii*'), recursive=True):
        print("Files exist. Skipping {out_dir_indep_var}")
        continue
    
    from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import RegressionNPYPreparer
    preparer = RegressionNPYPreparer(
        design_matrix=design_matrix,
        contrast_matrix=contrast_matrix_df,
        outcome_matrix=outcome_matrix,
        out_dir=out_dir_indep_var,
        mask_path=mask_path,
        exchangeability_blocks=None,   # or your DataFrame
        data_transform_method=data_transform_method
    )
    dataset_dict, json_path = preparer.run()

    from calvin_utils.neuroimaging_utils.ccm_utils.npy_regression import RegressionNPYAnalysis
    reg = RegressionNPYAnalysis(data_dict_path=json_path,
                        n_permutations=1000, 
                        out_dir=out_dir_indep_var,
                        fwe=True,
                        max_stat_method="pseudo_var_smooth",
                        mask_path=mask_path,
                        verbose=False)
    results = reg.run()
    print("Voxelwise FWE p-values shape:", results["voxelwise_p_values"].shape)

    # Save and visualize results
    reg.save_and_visualize_results(verbose=False)  # Change to False to disable visualization
