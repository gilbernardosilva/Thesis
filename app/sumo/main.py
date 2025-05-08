import argparse
import logging
import os
import sys
from datetime import datetime

os.environ["LOKY_MAX_CPU_COUNT"] = "14"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    from app.test.sumo_generator import generate_sumo_files
    from app.test.ai_pipeline import ai_pipeline
except ImportError as e:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.error(f"Erro ao importar módulos: {str(e)}")
    logger.error(
        "Certifique-se de que sumo_generator.py e ai_pipeline.py estão no diretório app/test."
    )
    sys.exit(1)


def setup_logging(base_path, date_folder, subfolder):
    output_dir = os.path.join(base_path, "output", date_folder, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "main.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def main(args):
    base_path = os.path.abspath(args.base_path)
    date_folder = datetime.now().strftime("%Y-%m-%d")
    should_generate = args.all_steps or args.all_steps_optimize or args.generate
    should_analyze = args.all_steps or args.all_steps_optimize or args.analyze

    # Define os caminhos padrão
    default_fcd_dir = os.path.join(base_path, "output", "default", "fcd")
    default_trips_dir = os.path.join(base_path, "output", "default", "trips")
    os.makedirs(default_fcd_dir, exist_ok=True)
    os.makedirs(default_trips_dir, exist_ok=True)

    if should_generate:
        logger = setup_logging(base_path, date_folder, "SUMO")
        logger.info(
            f"Iniciando execução com base_path: {base_path} e data: {date_folder}"
        )
        logger.info("Gerando novos cenários SUMO...")
        fcd_files = generate_sumo_files(args, logger, date_folder)
        if not fcd_files:
            logger.error("Falha ao gerar arquivos SUMO. Encerrando.")
            return
        logger.info(f"Arquivos SUMO gerados: {fcd_files}")
    else:
        logger = setup_logging(base_path, date_folder, "PIPELINE")

    if should_analyze:
        logger.info("Executando pipeline de IA...")
        # Usa a pasta da data atual se gerou SUMO, senão usa default
        fcd_dir = (
            os.path.join(base_path, "output", date_folder, "SUMO", "fcd")
            if should_generate
            else default_fcd_dir
        )
        if not os.path.exists(fcd_dir) or not any(
            f.endswith(".xml") for f in os.listdir(fcd_dir)
        ):
            logger.error(
                f"Nenhum arquivo FCD encontrado em {fcd_dir}. Use --generate, --all-steps ou --all-steps-optimize primeiro."
            )
            return

        if args.all_steps_optimize:
            logger.info("Executando pipeline com otimização bayesiana ativada")
            args.optimize = True
        else:
            args.optimize = False

        results, df = ai_pipeline(
            args, logger, date_folder if should_generate else "default"
        )
        if results is None or df is None:
            logger.error("Falha ao executar o pipeline de IA.")
            return
        logger.info("Pipeline de IA concluído com sucesso!")
        f1_scores = {
            k: v["metrics"].get("scenario4", {}).get("f1", 0)
            for k, v in results.items()
        }
        logger.info(f"Resultados (F1-score do scenario4): {f1_scores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Controle de geração de cenários SUMO e análise de IA"
    )
    parser.add_argument(
        "--base-path",
        default=os.getcwd(),
        help="Diretório base para os arquivos (raiz do projeto)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Gera novos cenários SUMO (roda sumo_generator.py)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Executa o pipeline de IA (roda ai_pipeline.py)",
    )
    parser.add_argument(
        "--use-processed-data",
        action="store_true",
        help="Usa dados pré-processados em CSV, se disponíveis (apenas com --analyze ou --all-steps)",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Executa todos os passos: gera cenários SUMO e executa o pipeline de IA (sem otimização)",
    )
    parser.add_argument(
        "--all-steps-optimize",
        action="store_true",
        help="Executa todos os passos: gera cenários SUMO e executa o pipeline de IA com otimização bayesiana",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Ativa a otimização bayesiana no pipeline de IA (usado com --analyze ou sobrescrito por --all-steps-optimize)",
    )

    args = parser.parse_args()

    if not (args.generate or args.analyze or args.all_steps or args.all_steps_optimize):
        date_folder = datetime.now().strftime("%Y-%m-%d")
        logger = setup_logging(args.base_path, date_folder, "PIPELINE")
        logger.error(
            "Nenhuma ação especificada. Use --generate, --analyze, --all-steps, --all-steps-optimize ou uma combinação."
        )
        parser.print_help()
        sys.exit(1)

    main(args)
