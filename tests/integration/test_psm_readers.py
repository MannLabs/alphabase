"""Integration tests for the PSM Readers.

Tests the output of defined inputs against reference data, which are expected in the `reference_data` folder.

Most of the test data is taken from psm_readers.ipynb
"""

import io
import os
from io import StringIO
from pathlib import Path

import pandas as pd

from alphabase.psm_reader import (
    DiannReader,
    MaxQuantReader,
    SpectronautReader,
    SpectronautReportReader,
    SwathReader,
    pFindReader,
    psm_reader_yaml,
)
from alphabase.psm_reader.keys import LibPsmDfCols, PsmDfCols
from alphabase.spectral_library.reader import LibraryReaderBase

current_file_directory = os.path.dirname(os.path.abspath(__file__))
test_data_path = Path(f"{current_file_directory}/reference_data")

# TODO add tests for AlphaPept


def _assert_reference_df_equal(psm_df: pd.DataFrame, test_case_name: str) -> None:
    """Compare the output of a PSM reader against reference data.

    If reference is not present, save the output as reference data and raise.
    """
    out_file_path = test_data_path / f"reference_{test_case_name}.parquet"
    # psm_df.to_csv(test_data_path / f"reference_{test_case_name}.csv")

    # check that all columns are available in PsmDfCols
    assert (
        set(psm_df.columns)
        - set(PsmDfCols.get_values())
        - set(LibPsmDfCols.get_values())
        == set()
    )

    if out_file_path.exists():
        expected_df = pd.read_parquet(out_file_path)

        pd.testing.assert_frame_equal(psm_df, expected_df)
    else:
        psm_df.to_parquet(out_file_path)
        raise ValueError("No reference data found.")


def test_psm_reader_yaml() -> None:
    """Test that all column mappings in the psm_reader.yaml are covered by string constant keys."""
    for reader_config in psm_reader_yaml.values():
        ks = [k for k in reader_config["column_mapping"]]
        assert (
            set(ks) - set(PsmDfCols.get_values()) - set(LibPsmDfCols.get_values())
            == set()
        )


def test_maxquant_reader() -> None:
    """Test the MaxQuant reader."""

    input_data = io.StringIO("""Raw file	Scan number	Scan index	Sequence	Length	Missed cleavages	Modifications	Modified sequence	Oxidation (M) Probabilities	Oxidation (M) Score diffs	Acetyl (Protein_N-term)	Oxidation (M)	Proteins	Charge	Fragmentation	Mass analyzer	Type	Scan event number	Isotope index	m/z	Mass	Mass error [ppm]	Mass error [Da]	Simple mass error [ppm]	Retention time	PEP	Score	Delta score	Score diff	Localization prob	Combinatorics	PIF	Fraction of total spectrum	Base peak fraction	Precursor full scan number	Precursor Intensity	Precursor apex fraction	Precursor apex offset	Precursor apex offset time	Matches	Intensities	Mass deviations [Da]	Mass deviations [ppm]	Masses	Number of matches	Intensity coverage	Peak coverage	Neutral loss level	ETD identification type	Reverse	All scores	All sequences	All modified sequences	Reporter PIF	Reporter fraction	id	Protein group IDs	Peptide ID	Mod. peptide ID	Evidence ID	Oxidation (M) site IDs
    20190402_QX1_SeVW_MA_HeLa_500ng_LC11	81358	73979	AAAAAAAAAPAAAATAPTTAATTAATAAQ	29	0	Unmodified	_(Acetyl (Protein_N-term))AAAAAAAAM(Oxidation (M))PAAAATAPTTAATTAATAAQ_			0	0	sp|P37108|SRP14_HUMAN	3	HCD	FTMS	MULTI-MSMS	13	1	790.07495	2367.203	0.35311	0.00027898	-0.061634807	70.261	0.012774	41.423	36.666	NaN	NaN	1	0	0	0	81345	10653955	0.0338597821787898	-11	0.139877319335938	y1;y2;y3;y4;y11;y1-NH3;y2-NH3;a2;b2;b3;b4;b5;b6;b7;b8;b9;b11;b12;b6(2+);b8(2+);b13(2+);b18(2+)	2000000;2000000;300000;400000;200000;1000000;400000;300000;600000;1000000;2000000;3000000;3000000;3000000;3000000;2000000;600000;500000;1000000;2000000;300000;200000	5.2861228709844E-06;-6.86980268369553E-05;-0.00238178789771837;0.000624715964988809;-0.0145624692099773;-0.000143471782706683;-0.000609501446461991;-0.000524972720768346;0.00010190530804266;5.8620815195809E-05;0.000229901232955854;-0.000108750048696038;-0.000229593152369034;0.00183148682538103;0.00276641182404092;0.000193118923334623;0.00200988580445483;0.000102216846016745;5.86208151389656E-05;0.000229901232955854;-0.00104559184393338;0.00525030008475369	0.0359413365445091;-0.314964433555295;-8.23711898839045;1.60102421155213;-14.8975999917227;-1.10320467763838;-3.03102462870716;-4.56152475051625;0.712219104095465;0.273777366204575;0.806231096969562;-0.305312183824154;-0.537399178230218;3.67572664689217;4.85930954169285;0.301587577451224;2.48616190909398;0.116225745519871;0.273777365939099;0.806231096969562;-2.19774169175011;7.53961026980589	147.076413378177;218.113601150127;289.153028027798;390.197699998035;977.50437775671;130.050013034583;201.087592852046;115.087114392821;143.081402136892;214.118559209185;285.155501716567;356.192954155649;427.230188786552;498.265241494374;569.301420357176;640.341107437877;808.429168310795;879.468189767554;214.118559209185;285.155501716567;475.757386711244;696.362265007215	22	0.262893575628735	0.0826446280991736	None	Unknown		41.4230894199432;4.75668724862449;3.9515580701967	AAAAAAAAAPAAAATAPTTAATTAATAAQ;FHRGPPDKDDMVSVTQILQGK;PVTLWITVTHMQADEVSVWR	_AAAAAAAAAPAAAATAPTTAATTAATAAQ_;_FHRGPPDKDDMVSVTQILQGK_;_PVTLWITVTHMQADEVSVWR_			0	1443	0	0	0
    20190402_QX1_SeVW_MA_HeLa_500ng_LC11	81391	74010	AAAAAAAAAAPAAAATAPTTAATTAATAAQ	29	0	Unmodified	_AAAAAAAAAPAAAATAPTTAATTAATAAQ_			0	0	sp|P37108|SRP14_HUMAN	2	HCD	FTMS	MULTI-MSMS	14	0	1184.6088	2367.203	0.037108	4.3959E-05	1.7026696	70.287	7.1474E-09	118.21	100.52	NaN	NaN	1	0	0	0	81377	9347701	0.166790347889974	-10	0.12664794921875	y1;y2;y3;y4;y5;y9;y12;y13;y14;y20;y13-H2O;y20-H2O;y1-NH3;y20-NH3;b3;b4;b5;b6;b7;b8;b9;b11;b12;b13;b14;b15;b16;b19;b15-H2O;b16-H2O	500000;600000;200000;400000;200000;100000;200000;1000000;200000;300000;200000;100000;100000;70000;300000;900000;2000000;3000000;5000000;8000000;6000000;600000;800000;600000;200000;300000;200000;300000;300000;1000000	-0.000194444760495571;0.000149986878682284;0.000774202587820128;-0.0002445094036716;0.000374520568641401;-0.00694293246522193;-0.0109837291331587;-0.0037745820627606;-0.000945546471939451;0.00152326440706929;0.00506054832726477;0.00996886361417637;6.25847393393997E-05;-0.024881067836759;-3.11821549132674E-05;-0.000183099230639527;0.000161332473453513;0.000265434980121881;0.000747070697229901;0.000975534518261156;0.00101513939785036;0.00651913000274362;0.0058584595163893;0.00579536744021425;0.00131097834105276;-0.0131378531671089;0.00472955218901916;-0.00161006322559842;-0.00201443239325272;0.0227149399370319	-1.32206444236914;0.687655553213019;2.6775131607882;-0.626628140021726;0.811995006209331;-8.6203492854282;-10.1838066275079;-3.21078702288986;-0.758483069159249;0.881072738747222;4.37168212373889;5.82682888353564;0.481236695337485;-14.5343986203644;-0.145630261806375;-0.642102166533079;0.452935954800214;0.621293379181583;1.49934012872483;1.71355878380837;1.58531240493271;8.06399202403175;6.6614096214532;6.09718023739784;1.28333378040908;-11.7030234519348;3.96235146626144;-1.07856912288932;-1.82370619437775;19.3220953109188	147.07661310906;218.113382465221;289.149872037312;390.198569223404;461.235063981231;805.411965958065;1078.54847749073;1175.59403219566;1246.62831694787;1728.87474561429;1157.57463237897;1710.85573532879;130.049806978061;1711.87460084504;214.118649012155;285.155914717031;356.192684073126;427.22969375842;498.266325910503;569.303211234482;640.340285417402;808.424659066597;879.462433524883;950.49961040476;1021.54120858166;1122.60333588727;1193.62258226971;1492.77704268533;1104.58164778019;1175.59403219566	30	0.474003002083763	0.167630057803468	None	Unknown		118.209976573419;17.6937689289157;17.2534171481793	AAAAAAAAAPAAAATAPTTAATTAATAAQ;SELKQEAMQSEQLQSVLYLK;VGSSVPSKASELVVMGDHDAARR	_AAAAAAAAAPAAAATAPTTAATTAATAAQ_;_SELKQEAM(Oxidation (M))QSEQLQSVLYLK_;_VGSSVPSKASELVVMGDHDAARR_			1	1443	0	0	1
    20190402_QX1_SeVW_MA_HeLa_500ng_LC11	107307	98306	AAAAAAAGDSDSWDADAFSVEDPVRK	26	1	Acetyl (Protein_N-term)	_(Acetyl (Protein_N-term))AAAAAAAGDSDSWDADAFSVEDPVRK_			1	0	sp|O75822|EIF3J_HUMAN	3	HCD	FTMS	MULTI-MSMS	10	2	879.06841	2634.1834	-0.93926	-0.00082567	-3.2012471	90.978	2.1945E-12	148.95	141.24	NaN	NaN	1	0	0	0	107297	10193939	0.267970762043589	-8	0.10211181640625	y1;y2;y4;y5;y6;y7;y8;y9;y10;y11;y12;y13;y14;y15;y17;y18;y19;y20;y21;y23;y21-H2O;y1-NH3;y19-NH3;y14(2+);y16(2+);y22(2+);a2;b2;b3;b4;b5;b6;b7	300000;200000;3000000;600000;1000000;500000;2000000;1000000;1000000;1000000;90000;1000000;400000;900000;1000000;400000;3000000;2000000;1000000;400000;100000;200000;200000;80000;100000;200000;200000;2000000;5000000;5000000;5000000;2000000;300000	1.34859050149316E-07;-6.05140996867704E-06;2.27812602133781E-05;0.00128986659160546;-0.00934536073077652;0.000941953783126337;-0.00160424237344614;-0.00239257341399934;-0.00111053968612396;-0.00331340710044969;0.00330702864630439;0.000963683996815234;0.00596290290945944;-0.00662057038289277;-0.0117122701335575;0.00777853472800416;0.0021841542961738;0.000144322111736983;-0.00087403893667215;0.0197121595674616;-0.021204007680808;-0.000308954599830713;-0.026636719419912;-0.0137790992353075;0.00596067266928912;-0.0077053835773313;9.11402199221811E-06;-0.000142539300128419;-0.000251999832926231;1.90791054137662E-05;-0.00236430185879044;-9.54583337602344E-05;-0.000556959493223985	0.000916705048437201;-0.0199575598103408;0.0456231928690862;2.09952637717462;-12.5708704058425;1.11808305811426;-1.72590731777249;-2.22239181008062;-0.967696370445928;-2.62418809422166;2.47964286628144;0.665205752892023;3.64753748704453;-3.84510115530963;-6.08782672045773;3.81508105974837;1.04209904973991;0.0666012719936656;-0.390545453668809;8.28224925531311;-9.55133250134922;-2.37499239179248;-12.8127653858411;-16.846761946123;6.48662354975264;-6.67117082062383;0.0580151981289049;-0.770098855873447;-0.983876895688683;0.0583162347158579;-5.93738717724506;-0.203431522818505;-1.03087538746314	147.112804035741;303.21392125011;499.33507018564;614.360746132308;743.413974455831;842.472101057517;929.506675663573;1076.57587791081;1147.61170966489;1262.6408555643;1333.67134891635;1448.700635293;1634.77494902759;1721.81956091078;1923.88362405243;2038.89107627957;2095.9181343836;2166.95728800359;2237.99542015244;2380.04906152953;2220.00518543488;130.0865640237;2078.92040615582;817.907873297785;918.917619246831;1155.02717356753;157.097144992378;185.0922112678;256.129434516133;327.166277224995;398.205774393759;469.240619338034;540.278194626993	33	0.574496146107112	0.14410480349345	None	Unknown		148.951235201399;7.71201258444522;7.36039532447559	AAAAAAAGDSDSWDADAFSVEDPVRK;PSRQESELMWQWVDQRSDGER;HTLTSFWNFKAGCEEKCYSNR	_(Acetyl (Protein_N-term))AAAAAAAGDSDSWDADAFSVEDPVRK_;_PSRQESELM(Oxidation (M))WQWVDQRSDGER_;_HTLTSFWNFKAGCEEKCYSNR_			2	625	1	1	2	""")

    reader = MaxQuantReader()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "maxquant")


def test_pfind_reader() -> None:
    """Test the pFind reader."""
    input_data = StringIO("""File_Name	Scan_No	Exp.MH+	Charge	Q-value	Sequence	Calc.MH+	Mass_Shift(Exp.-Calc.)	Raw_Score	Final_Score	Modification	Specificity	Proteins	Positions	Label	Target/Decoy	Miss.Clv.Sites	Avg.Frag.Mass.Shift	Others
    Ecoli-1to1to1-un-C13-N15-10mM-20150823.30507.30507.2.0.dta	30507	2074.030369	2	0	AMIEAGAAAVHFEDQLASVK	2074.027271	0.003098	35.299588	5.15726e-013	2,Oxidation[M];	3	gi|16131841|ref|NP_418439.1|/	173,K,K/	1|0|	target	0	0.948977	131070	0	0	0	262143	0	0	0	32
    Ecoli-1to1to1-un-C13-N15-150mM-20150823.41501.41501.3.0.dta	41501	2712.197421	3	0	EGDNYVVLSDILGDEDHLGDMDFK	2712.198013	-0.000592	27.073978	9.82619e-010	21,Unknown[M];	3	gi|145698316|ref|NP_417633.4|/	470,K,V/	1|0|	target	0	0.814438	65596	0	0	0	4194288	0	0	0	36
    XXX.25802.25802.4.0.dta	25802	2388.339186	4	0.0032066	SVFLIKGDKVWVYPPEKKEK	2388.332468	0.006718	17.822784	0.100787	21,Didehydro[AnyC-termK];	0	sp|P02790|HEMO_HUMAN/	106,N,G/	1|0|	target	0	0.704714	36
    """)
    reader = pFindReader()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "pfind")


def test_diann_reader() -> None:
    """Test the Diann reader."""
    input_data = StringIO("""File.Name	Run	Protein.Group	Protein.Ids	Protein.Names	Genes	PG.Quantity	PG.Normalised	PG.MaxLFQ	Genes.Quantity	Genes.Normalised	Genes.MaxLFQ	Genes.MaxLFQ.Unique	Modified.Sequence	Stripped.Sequence	Precursor.Id	Precursor.Charge	Q.Value	Global.Q.Value	Protein.Q.Value	PG.Q.Value	Global.PG.Q.Value	GG.Q.Value	Translated.Q.Value	Proteotypic	Precursor.Quantity	Precursor.Normalised	Precursor.Translated	Quantity.Quality	RT	RT.Start	RT.Stop	iRT	Predicted.RT	Predicted.iRT	Lib.Q.Value	Ms1.Profile.Corr	Ms1.Area	Evidence	Spectrum.Similarity	Mass.Evidence	CScore	Decoy.Evidence	Decoy.CScore	Fragment.Quant.Raw	Fragment.Quant.Corrected	Fragment.Correlations	MS2.Scan	IM	iIM	Predicted.IM	Predicted.iIM
    F:\XXX\20201218_tims03_Evo03_PS_SA_HeLa_200ng_high_speed_21min_8cm_S2-A2_1_22636.d	20201218_tims03_Evo03_PS_SA_HeLa_200ng_high_speed_21min_8cm_S2-A2_1_22636	Q9UH36	Q9UH36		SRRD	3296.49	3428.89	3428.89	3296.49	3428.89	3428.89	3428.89	(UniMod:1)AAAAAAALESWQAAAPR	AAAAAAALESWQAAAPR	(UniMod:1)AAAAAAALESWQAAAPR2	2	3.99074e-05	1.96448e-05	0.000159821	0.000159821	0.000146135	0.000161212	0	1	3296.49	3428.89	3296.49	0.852479	19.9208	19.8731	19.9685	123.9	19.8266	128.292	0	0.960106	5308.05	1.96902	0.683134	0.362287	0.999997	1.23691	3.43242e-05	1212.01;2178.03;1390.01;1020.01;714.008;778.008;	1212.01;1351.73;887.591;432.92;216.728;732.751;	0.956668;0.757581;0.670497;0.592489;0.47072;0.855203;	30053	1.19708	1.19328	1.19453	1.19469
    F:\XXX\20201218_tims03_Evo03_PS_SA_HeLa_200ng_high_speed_21min_8cm_S2-A8_1_22642.d	20201218_tims03_Evo03_PS_SA_HeLa_200ng_high_speed_21min_8cm_S2-A8_1_22642	Q9UH36	Q9UH36		SRRD	2365	2334.05	2334.05	2365	2334.05	2334.05	2334.05	(UniMod:1)AAAAAAALESWQAAAPR	AAAAAAALESWQAAAPR	(UniMod:1)AAAAAAALESWQAAAPR2	2	0.000184434	1.96448e-05	0.000596659	0.000596659	0.000146135	0.000604961	0	1	2365	2334.05	2365	0.922581	19.905	19.8573	19.9527	123.9	19.782	128.535	0	0.940191	4594.04	1.31068	0.758988	0	0.995505	0.28633	2.12584e-06	1209.02;1210.02;1414.02;1051.01;236.003;130.002;	1209.02;1109.89;732.154;735.384;0;46.0967;	0.919244;0.937624;0.436748;0.639369;0.296736;0.647924;	30029	1.195	1.19328	1.19381	1.19339
    F:\XXX\20201218_tims03_Evo03_PS_SA_HeLa_200ng_high_speed_21min_8cm_S2-B2_1_22648.d	20201218_tims03_Evo03_PS_SA_HeLa_200ng_high_speed_21min_8cm_S2-B2_1_22648	Q9UH36	Q9UH36		SRRD	1664.51	1635.46	1635.47	1664.51	1635.46	1635.47	1635.47	(UniMod:1)AAAAAAALESWQAAAPR	AAAAAAALESWQAAAPR	(UniMod:1)AAAAAAALESWQAAAPR2	2	0.000185123	1.96448e-05	0.000307409	0.000307409	0.000146135	0.000311332	0	1	1664.51	1635.46	1664.51	0.811147	19.8893	19.8416	19.937	123.9	19.7567	128.896	0	0.458773	6614.06	1.7503	0.491071	0.00111683	0.997286	1.92753	2.80543e-05	744.01;1708.02;1630.02;1475.02;0;533.006;	322.907;808.594;577.15;536.033;0;533.006;	0.760181;0.764072;0.542005;0.415779;0;0.913438;	30005	1.19409	1.19328	1.19323	1.19308
    """)
    reader = DiannReader()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "diann")


def test_spectronaut_reader() -> None:
    """Test the Spectronaut reader."""
    input_data = StringIO("""ReferenceRun	PrecursorCharge	Workflow	IntModifiedPeptide	CV	AllowForNormalization	ModifiedPeptide	StrippedPeptide	iRT	IonMobility	iRTSourceSpecific	BGSInferenceId	IsProteotypic	IntLabeledPeptide	LabeledPeptide	PrecursorMz	ReferenceRunQvalue	ReferenceRunMS1Response	FragmentLossType	FragmentNumber	FragmentType	FragmentCharge	FragmentMz	RelativeIntensity	ExcludeFromAssay	Database	ProteinGroups	UniProtIds	Protein Name	ProteinDescription	Organisms	OrganismId	Genes	Protein Existence	Sequence Version	FASTAName
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_100ug_test_S4-A1_1_25843	2		_ALVAT[+80]PGK_		True	_ALVAT[Phospho (STY)]PGK_	ALVATPGK	-5.032703	0.758	-5.032703	P19338	False	_ALVAT[+80]PGK_	_ALVAT[Phospho (STY)]PGK_	418.717511324722	0	10352	noloss	3	y	1	301.187031733932	53.1991	False	sp	P19338	P19338	NUCL_HUMAN	Nucleolin	Homo sapiens		NCL	1	3	MCT_human_UP000005640_9606
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_100ug_test_S4-A1_1_25843	2		_ALVAT[+80]PGK_		True	_ALVAT[Phospho (STY)]PGK_	ALVATPGK	-5.032703	0.758	-5.032703	P19338	False	_ALVAT[+80]PGK_	_ALVAT[Phospho (STY)]PGK_	418.717511324722	0	10352	H3PO4	4	y	1	384.224142529733	26.31595	False	sp	P19338	P19338	NUCL_HUMAN	Nucleolin	Homo sapiens		NCL	1	3	MCT_human_UP000005640_9606
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_100ug_test_S4-A1_1_25843	2		_TLT[+80]PCPLR_		True	_TLT[Phospho (STY)]PC[Carbamidomethyl (C)]PLR_	TLTPCPLR	27.71659	0.818	27.71659	Q5T200	False	_TLT[+80]PPLR_	_TLT[Phospho (STY)]PPLR_	439.230785875227	0.000138389150379226	23117	noloss	3	b	1	396.153027901512	6.3264	False	sp	Q5T200	Q5T200	ZC3HD_HUMAN	Zinc finger CCCH domain-containing protein 13	Homo sapiens		ZC3H13	1	1	MCT_human_UP000005640_9606
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_100ug_test_S4-A1_1_25843	2		_TLT[+80]PCPLR_		True	_TLT[Phospho (STY)]PC[Carbamidomethyl (C)]PLR_	TLTPCPLR	27.71659	0.818	27.71659	Q5T200	False	_TLT[+80]PPLR_	_TLT[Phospho (STY)]PPLR_	439.230785875227	0.000138389150379226	23117	noloss	3	y	1	385.255780000092	29.70625	False	sp	Q5T200	Q5T200	ZC3HD_HUMAN	Zinc finger CCCH domain-containing protein 13	Homo sapiens		ZC3H13	1	1	MCT_human_UP000005640_9606
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_library25_S4-C1_1_25867	2		_LFVT[+80]PPEGSSR_		True	_[Acetyl (Protein_N-term)]LFVS[Phospho (STY)]PPEGSSR_	LFVSPPEGSSR	38.05031	0.917	38.05031	Q14244;Q14244-6;Q14244-7	False	_LFVT[+80]PPEGSSR_	_LFVT[Phospho (STY)]PPEGSSR_	635.297385373987	0	14164	H3PO4	4	b	1	443.265279065723	12.24525	False	sp	Q14244;Q14244-6;Q14244-7	Q14244;Q14244-6;Q14244-7	MAP7_HUMAN	Ensconsin;Isoform of Q14244, Isoform 6 of Ensconsin;Isoform of Q14244, Isoform 7 of Ensconsin	Homo sapiens		MAP7	1;;	1;;	MCT_human_UP000005640_9606;MCT_human2_UP000005640_9606_additional;MCT_human2_UP000005640_9606_additional
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_library25_S4-C1_1_25867	2		_LFVT[+80]PPEGSSR_		True	_[Acetyl (Protein_N-term)]LFVS[Phospho (STY)]PPEGSSR_	LFVSPPEGSSR	38.05031	0.917	38.05031	Q14244;Q14244-6;Q14244-7	False	_LFVT[+80]PPEGSSR_	_LFVT[Phospho (STY)]PPEGSSR_	635.297385373987	0	14164	noloss	6	y	1	632.299829640042	46.07855	False	sp	Q14244;Q14244-6;Q14244-7	Q14244;Q14244-6;Q14244-7	MAP7_HUMAN	Ensconsin;Isoform of Q14244, Isoform 6 of Ensconsin;Isoform of Q14244, Isoform 7 of Ensconsin	Homo sapiens		MAP7	1;;	1;;	MCT_human_UP000005640_9606;MCT_human2_UP000005640_9606_additional;MCT_human2_UP000005640_9606_additional
    202106018_TIMS03_EVO03_PaSk_SA_HeLa_EGF_Phospho_library25_S4-C1_1_25867	2		_LFVT[+80]PPEGSSR_		True	_[Acetyl (Protein_N-term)]LFVS[Phospho (STY)]PPEGSSR_	LFVSPPEGSSR	38.05031	0.917	38.05031	Q14244;Q14244-6;Q14244-7	False	_LFVT[+80]PPEGSSR_	_LFVT[Phospho (STY)]PPEGSSR_	635.297385373987	0	14164	noloss	7	y	1	729.352593488892	100	False	sp	Q14244;Q14244-6;Q14244-7	Q14244;Q14244-6;Q14244-7	MAP7_HUMAN	Ensconsin;Isoform of Q14244, Isoform 6 of Ensconsin;Isoform of Q14244, Isoform 7 of Ensconsin	Homo sapiens		MAP7	1;;	1;;	MCT_human_UP000005640_9606;MCT_human2_UP000005640_9606_additional;MCT_human2_UP000005640_9606_additional
    """)

    reader = SpectronautReader()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "spectronaut")


def test_spectronaut_report_reader() -> None:
    """Test the Spectronaut report reader."""
    input_data = StringIO("""R.FileName,R.Replicate,EG.PrecursorId,EG.ApexRT,FG.CalibratedMassAccuracy (PPM),FG.CalibratedMz
    20211203_EXPL2_SoSt_SA_DIA_HeLa_1000mz_noCB_01,1,_VIETPENDFK_.2,40.826847076416,-0.6350307649846,596.298998773218
    20211203_EXPL2_SoSt_SA_DIA_HeLa_1000mz_noCB_01,1,_GFSNEVSSK_.2,19.1254806518555,-1.54873822486555,477.730400257423
    20211203_EXPL2_SoSt_SA_DIA_HeLa_1000mz_noCB_01,1,_HLLNQAVGEEEVPK_.3,42.0593299865723,-0.309173676987587,521.611288926824
    20211203_EXPL2_SoSt_SA_DIA_HeLa_1000mz_noCB_01,1,_DATM[Oxidation (M)]EVQR_.2,12.8398199081421,-3.31103772642203,483.222124398527
    """)

    reader = SpectronautReportReader()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "spectronaut_report")


def test_openswath_reader() -> None:
    """Test the OpenSwath reader."""

    input_data = StringIO("""PrecursorMz	ProductMz	Tr_recalibrated	transition_name	CE	LibraryIntensity	transition_group_id	decoy	PeptideSequence	ProteinName	Annotation	FullUniModPeptideName	PrecursorCharge	GroupLabel	UniprotID	FragmentType	FragmentCharge	FragmentSeriesNumber
    685.732240417	886.020494795	59.0	255_AAAAAAAAAASGAAIPPLIPPRR_3	-1	5257.9	13_AAAAAAAAAASGAAIPPLIPPRR_3	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	y19^2/0.002	AAAAAAAAAASGAAIPPLIPPRR	3	light	1/O14654	y	2	19
    514.550999438	473.303261576	59.2	268_AAAAAAAAAASGAAIPPLIPPRR_4	-1	10000.0	14_AAAAAAAAAASGAAIPPLIPPRR_4	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	y8^2/0.002	AAAAAAAAAASGAAIPPLIPPRR	4	light	1/O14654	y	2	8
    514.550999438	629.39313922	59.2	276_AAAAAAAAAASGAAIPPLIPPRR_4	-1	5923.1	14_AAAAAAAAAASGAAIPPLIPPRR_4	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	y12^2/0.001	AAAAAAAAAASGAAIPPLIPPRR	4	light	1/O14654	y	2	12
    514.550999438	672.909153425	59.2	279_AAAAAAAAAASGAAIPPLIPPRR_4	-1	5249.8	14_AAAAAAAAAASGAAIPPLIPPRR_4	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	y13^2/0.001	AAAAAAAAAASGAAIPPLIPPRR	4	light	1/O14654	y	2	13
    514.550999438	356.19284545	59.2	262_AAAAAAAAAASGAAIPPLIPPRR_4	-1	5233.6	14_AAAAAAAAAASGAAIPPLIPPRR_4	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	b5/0.001,b10^2/0.001,m6:10/0.001	AAAAAAAAAASGAAIPPLIPPRR	4	light	1/O14654	b	1	5
    514.550999438	498.26707303	59.2	269_AAAAAAAAAASGAAIPPLIPPRR_4	-1	4976.0	14_AAAAAAAAAASGAAIPPLIPPRR_4	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	b7/0.001,m4:10/0.001	AAAAAAAAAASGAAIPPLIPPRR	4	light	1/O14654	b	1	7
    514.550999438	427.22995924	59.2	265_AAAAAAAAAASGAAIPPLIPPRR_4	-1	4859.4	14_AAAAAAAAAASGAAIPPLIPPRR_4	0	AAAAAAAAAASGAAIPPLIPPRR	1/O14654	b6/0.002,m5:10/0.002	AAAAAAAAAASGAAIPPLIPPRR	4	light	1/O14654	b	1	6
    728.201724416	356.19284545	101.8	292_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	-1	10000.0	15_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	0	AAAAAAAAAASGAAIPPLIPPRRVITLYQCFSVSQR	1/O14654	b5/0.003,b10^2/0.003,m6:10/0.003	AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR	5	light	1/O14654	b	1	5
    728.201724416	576.310000482	101.8	297_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	-1	7611.0	15_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	0	AAAAAAAAAASGAAIPPLIPPRRVITLYQCFSVSQR	1/O14654	y5/0.002	AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR	5	light	1/O14654	y	1	5
    728.201724416	427.22995924	101.8	293_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	-1	6805.1	15_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	0	AAAAAAAAAASGAAIPPLIPPRRVITLYQCFSVSQR	1/O14654	b6/-0.002,m5:10/-0.002	AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR	5	light	1/O14654	b	1	6
    728.201724416	569.30418682	101.8	296_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	-1	6312.7	15_AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR_5	0	AAAAAAAAAASGAAIPPLIPPRRVITLYQCFSVSQR	1/O14654	b8/0.009,m3:10/0.009	AAAAAAAAAASGAAIPPLIPPRRVITLYQC(UniMod:4)FSVSQR	5	light	1/O14654	b	1	8
    """)

    reader = SwathReader()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "openswath")


def test_diann_speclib_reader() -> None:
    """Test the Diann speclib reader."""
    # this is the head of  "https://datashare.biochem.mpg.de/s/DF12ObSdZnBnqUV" ("diann_speclib.tsv")
    input_data = StringIO("""FileName	PrecursorMz	ProductMz	Tr_recalibrated	IonMobility	transition_name	LibraryIntensity	transition_group_id	decoy	PeptideSequence	Proteotypic	QValue	PGQValue	Ms1ProfileCorr	ProteinGroup	ProteinName	Genes	FullUniModPeptideName	ModifiedPeptide	PrecursorCharge	PeptideGroupLabel	UniprotID	NTerm	CTerm	FragmentType	FragmentCharge	FragmentSeriesNumber	FragmentLossType	ExcludeFromAssay
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	645.36896	-17.011904	0	AAAAAAAAAVSR2_121_1_0_5	1	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	7	noloss	False
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	716.40607	-17.011904	0	AAAAAAAAAVSR2_121_1_0_4	0.92588264	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	8	noloss	False
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	574.33185	-17.011904	0	AAAAAAAAAVSR2_121_1_0_6	0.73629588	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	6	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	503.29471	-17.011904	0	AAAAAAAAAVSR2_121_1_0_7	0.47699517	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	5	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	214.11917	-17.011904	0	AAAAAAAAAVSR2_98_1_0_3	0.47343451	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	b	1	3	noloss	False
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	787.44318	-17.011904	0	AAAAAAAAAVSR2_121_1_0_3	0.39700398	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	9	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	285.15628	-17.011904	0	AAAAAAAAAVSR2_98_1_0_4	0.30815825	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	b	1	4	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	432.2576	-17.011904	0	AAAAAAAAAVSR2_121_1_0_8	0.26575705	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	4	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	356.19339	-17.011904	0	AAAAAAAAAVSR2_98_1_0_5	0.23726191	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	b	1	5	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	858.48029	-17.011904	0	AAAAAAAAAVSR2_121_1_0_2	0.23109815	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	10	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	427.2305	-17.011904	0	AAAAAAAAAVSR2_98_1_0_6	0.13046893	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	b	1	6	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral3/2023_12/20231213_OA3_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P27_70091_cb_H6_1424.mzML	500.78116	361.22049	-17.011904	0	AAAAAAAAAVSR2_121_1_0_9	0.11459313	AAAAAAAAAVSR2	0	AAAAAAAAAVSR	0	1.4398672e-05	0.002044061	0.63356501	Q96JP5;Q96JP5-2	ZFP91-2_HUMAN;ZFP91_HUMAN	ZFP91	AAAAAAAAAVSR	AAAAAAAAAVSR	2	AAAAAAAAAVSR	Q96JP5;Q96JP5-2;A0A0A6YYC7	0	0	y	1	3	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	672.40503	-14.478184	0	AAAAAAALQAK2_121_1_0_4	1	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	7	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	601.36786	-14.478184	0	AAAAAAALQAK2_121_1_0_5	0.81051117	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	6	noloss	False
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	214.11917	-14.478184	0	AAAAAAALQAK2_98_1_0_3	0.6025809	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	b	1	3	noloss	False
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	743.44214	-14.478184	0	AAAAAAALQAK2_121_1_0_3	0.55991524	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	8	noloss	False
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	530.33075	-14.478184	0	AAAAAAALQAK2_121_1_0_6	0.42974085	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	5	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	814.47925	-14.478184	0	AAAAAAALQAK2_121_1_0_2	0.40478998	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	9	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	285.15628	-14.478184	0	AAAAAAALQAK2_98_1_0_4	0.27873126	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	b	1	4	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	459.29367	-14.478184	0	AAAAAAALQAK2_121_1_0_7	0.23921044	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	4	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	346.20959	-14.478184	0	AAAAAAALQAK2_121_1_0_8	0.17267427	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	3	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	356.19339	-14.478184	0	AAAAAAALQAK2_98_1_0_5	0.11922429	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	b	1	5	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	427.2305	-14.478184	0	AAAAAAALQAK2_98_1_0_6	0.042955909	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	b	1	6	noloss	True
/fs/pool/pool-mann-ms14/MZML/Astral2/2023_12/20231206_OA2_ViAl_SA_FAIMS40_IO4_A556_MOMI-20231121_APAK_P35_72214_cb_A9_926.mzML	478.78064	885.51636	-14.478184	0	AAAAAAALQAK2_121_1_0_1	0.019872207	AAAAAAALQAK2	0	AAAAAAALQAK	1	1.5083094e-06	0.00029770765	0.97250301	P36578	RL4_HUMAN	RPL4	AAAAAAALQAK	AAAAAAALQAK	2	AAAAAAALQAK	P36578;H3BU31;H3BM89	0	0	y	1	10	noloss	True
    """)

    reader = LibraryReaderBase()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "diann_speclib")


def test_msfragger_speclib_reader() -> None:
    """Test the MSFragger speclib reader."""

    # this is the head of https://datashare.biochem.mpg.de/s/Cka1utORt3r5A4a ("msfragger_speclib.tsv")
    input_data = StringIO("""ModifiedPeptide	PrecursorCharge	Tr_recalibrated	IonMobility	StrippedPeptide	PrecursorMz	ProteinID	Genes	FragmentType	FragmentMz	RelativeIntensity	FragmentCharge	FragmentNumber	FragmentLossType
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	547.3086	1.0	1	5	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	660.39264	0.51416683	1	6	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	213.15976	0.2875934	1	2	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	305.18195	0.28558257	1	3	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	418.26602	0.22692133	1	4	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	204.13426	0.14408894	1	2	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	330.69998	0.047275875	2	6	noloss
_VLELTGK_	2	4.249923	0.75	VLELTGK	380.234178887762	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	342.20236	0.039677892	1	3	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	305.18195	1.0	1	3	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	418.26602	0.7783308	1	4	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	342.20236	0.7754817	1	3	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	547.3086	0.5358066	1	5	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	556.3341	0.38568112	1	5	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	455.2864	0.2794164	1	4	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	213.15976	0.25726682	1	2	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	204.13426	0.19917944	1	2	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	y	660.39264	0.12131426	1	6	noloss
_VLELTGK_	1	4.249923	1.284	VLELTGK	759.4610813087119	A0A0B4J2D5;P0DPI2	GATD3B;GATD3	b	613.3555	0.11152817	1	6	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	269.67664	1.0	2	4	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	326.21866	0.9999356	2	5	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	538.346	0.90463305	1	4	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	401.28708	0.71957177	1	3	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	651.43005	0.6489045	1	5	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	b	229.11829	0.4481002	1	2	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	288.203	0.28588438	1	2	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	y	375.75287	0.16287889	2	6	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	b	479.26126	0.09841085	1	4	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	b	592.34534	0.06371137	1	5	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	b	342.20236	0.030663786	1	3	noloss
_EVLHLLR_	2	21.775629	0.837	EVLHLLR	440.274169715862	A0AVF1	TTC26	b	705.4294	0.02848413	1	6	noloss
""")

    reader = LibraryReaderBase()
    reader.import_file(input_data)

    _assert_reference_df_equal(reader.psm_df, "msfragger_speclib")