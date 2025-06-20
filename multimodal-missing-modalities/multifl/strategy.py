from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Optional, Union, List, Tuple, Dict
from logging import INFO, WARNING, ERROR
from flwr.server.client_manager import ClientManager
from collections import OrderedDict
import json
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Code
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from multifl.utils import get_modality, set_modality


# Constants for layer prefixes
IMAGE_PREFIX = "encoders.image"
TEXT_PREFIX = "encoders.text"
FUSION_PREFIX = "fusion_module."
HEAD_PREFIX = "head_module."


class FedSim(FedAvg):
    def __init__(self, parameter_keys: List[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_parameters: Optional[Parameters] = None
        if not parameter_keys:
             raise ValueError("FedSim requires a non-empty list of parameter_keys.")
        self._parameter_keys: List[str] = parameter_keys
        self.client_modalities: Dict[str, int] = {}

    def initialize_modalities(self, client_manager : ClientManager, num_modalities: int = 3):
        all_clients = list(client_manager.all().values())
        sorted_clients = sorted(all_clients, key=lambda c: c.cid)
        for idx, client in enumerate(sorted_clients):
            modality = idx % num_modalities
            self.client_modalities[client.cid] = modality
            set_modality(client.cid, modality)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters:
            self._current_parameters = initial_parameters
        log(INFO, "Using initial parameters provided by base strategy or client.")
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure training: Round 1 gets modality, later rounds train."""
        if not self.client_modalities:
            self.initialize_modalities(client_manager)
        
        self._current_parameters = parameters # Store for aggregation

        if self._parameter_keys is None:
             log(ERROR, "Parameter keys not set. Cannot proceed.")
             return []

        base_config = {}
        if self.on_fit_config_fn is not None:
            base_config = self.on_fit_config_fn(server_round)

        base_config["server_round"] = server_round

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_ins_list = []
        full_ndarrays = parameters_to_ndarrays(parameters)
        full_named_ndarrays = OrderedDict(zip(self._parameter_keys, full_ndarrays))

        log(INFO, f"Round {server_round}: Configuring clients for training.")
        configured_clients = 0
        for client in clients:
            modality = get_modality(client.cid)
            client_config = base_config.copy()
            client_config["modality"] = modality

            if modality is None:
                log(WARNING, f"modality for client {client.cid} unknown in round {server_round}. Skipping.")
                continue

            client_named_ndarrays = OrderedDict()
            for name, param in full_named_ndarrays.items():
                send_param = False
                is_image = name.startswith(IMAGE_PREFIX)
                is_text = name.startswith(TEXT_PREFIX)
                is_fusion = name.startswith(FUSION_PREFIX)
                is_head = name.startswith(HEAD_PREFIX)

                if modality == 0:
                    if is_image or is_text or is_fusion or is_head: send_param = True
                elif modality == 1:
                    if is_text or is_fusion or is_head: send_param = True
                elif modality == 2:
                    if is_image or is_fusion or is_head: send_param = True

                if send_param:
                    client_named_ndarrays[name] = param
            
            if not client_named_ndarrays:
                log(WARNING, f"No parameters selected for client {client.cid} (modality {modality}). Skipping.")
                continue

            param_keys = list(client_named_ndarrays.keys())
            client_ndarrays = list(client_named_ndarrays.values())
            client_parameters = ndarrays_to_parameters(client_ndarrays)

            client_config["param_keys_json"] = json.dumps(param_keys)

            fit_ins = FitIns(client_parameters, client_config)
            fit_ins_list.append((client, fit_ins))
            configured_clients += 1

        log(INFO, f"Configured {configured_clients} clients for round {server_round} training.")
        return fit_ins_list


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation: Send full parameters and all keys as JSON."""
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        num_available = client_manager.num_available()
        if num_available < self.min_evaluate_clients:
             log(WARNING, f"Not enough clients available for evaluation ({num_available} < {self.min_evaluate_clients}). Skipping evaluation round.")
             return []

        sample_size = int(self.fraction_evaluate * num_available)
        sample_size = max(sample_size, self.min_evaluate_clients)

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients
        )

        evaluate_ins_list = []
        config["param_keys_json"] = json.dumps(self._parameter_keys)

        for client in clients:
            # log(INFO, f"Configuring evaluation for client {client.cid}")
            evaluate_ins = EvaluateIns(parameters, config.copy())
            evaluate_ins_list.append((client, evaluate_ins))

        log(INFO, f"Configured {len(evaluate_ins_list)} clients for evaluation round {server_round}")
        return evaluate_ins_list


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters, storing modality."""
        if not results:
            log(WARNING, "aggregate_fit: no results received.")
            return self._current_parameters, {}
        if not self.accept_failures and failures:
             log(WARNING, "aggregate_fit: failures received and accept_failures is False.")
             return self._current_parameters, {}

        for client, fit_res in results:
            if fit_res.status.code != Code.OK:
                 log(WARNING, f"aggregate_fit: Ignoring failed FitRes from client {client.cid} (status: {fit_res.status})")
                 continue

        if self._current_parameters is None:
             log(ERROR, "aggregate_fit: current_parameters not set.")
             return None, {}
        if self._parameter_keys is None:
             log(ERROR, "aggregate_fit: server parameter keys not set.")
             return None, {}

        current_global_ndarrays = parameters_to_ndarrays(self._current_parameters)
        global_named_parameters = OrderedDict(zip(self._parameter_keys, current_global_ndarrays))

        weights_results_full_structure: List[Tuple[NDArrays, int]] = []
        total_examples = 0

        # Filter results to only include successful ones for aggregation
        successful_results = [(client, res) for client, res in results if res.status.code == Code.OK]

        for client, fit_res in successful_results:
            client_ndarrays_partial = parameters_to_ndarrays(fit_res.parameters)

            if not client_ndarrays_partial:
                    log(WARNING, f"Client {client.cid} sent empty parameters in round {server_round}. Skipping.")
                    continue
            
            modality = get_modality(client.cid)
            print(f"Getting modality for: {client.cid} -> {get_modality(client.cid)}")
            if modality is None:
                log(WARNING, f"Modality for client {client.cid} unknown during aggregation in round {server_round}. Skipping.")
                continue

            client_keys = []
            for name in self._parameter_keys:
                should_have_key = False
                is_image = name.startswith(IMAGE_PREFIX)
                is_text = name.startswith(TEXT_PREFIX)
                is_fusion = name.startswith(FUSION_PREFIX)
                is_head = name.startswith(HEAD_PREFIX)

                if modality == 0: # Both
                    if is_image or is_text or is_fusion or is_head: should_have_key = True
                elif modality == 1: # Text only
                    if is_text or is_fusion or is_head: should_have_key = True
                elif modality == 2: # Image only
                    if is_image or is_fusion or is_head: should_have_key = True

                if should_have_key:
                    client_keys.append(name)

            if len(client_keys) != len(client_ndarrays_partial):
                log(WARNING, f"Client {client.cid} returned mismatch between keys ({len(client_keys)}) and parameters ({len(client_ndarrays_partial)}). Skipping.")
                continue

            client_full_ndarrays = []
            client_updates = dict(zip(client_keys, client_ndarrays_partial))

            valid_structure = True
            for key in self._parameter_keys:
                if key in client_updates:
                    client_full_ndarrays.append(client_updates[key])
                else:
                    # Append the global parameter if the client didn't send this key
                    if key in global_named_parameters:
                        modality = get_modality(client.cid)
                        should_have_key = False
                        is_image = key.startswith(IMAGE_PREFIX)
                        is_text = key.startswith(TEXT_PREFIX)
                        is_fusion = key.startswith(FUSION_PREFIX)
                        is_head = key.startswith(HEAD_PREFIX) # Head is always expected

                        if modality == 0: # Both
                            if is_image or is_text or is_fusion or is_head: should_have_key = True
                        elif modality == 1: # Text only
                            if is_text or is_fusion or is_head: should_have_key = True
                        elif modality == 2: # Image only
                            if is_image or is_fusion or is_head: should_have_key = True

                        # If the key should have been sent but wasn't, log an error.
                        # Using the global parameter is the intended fallback here.
                        if should_have_key:
                            log(ERROR, f"Client {client.cid} (modality {modality}) was expected to send key '{key}' but did not. Using previous global value for this key during aggregation reconstruction.")

                        client_full_ndarrays.append(global_named_parameters[key])
                    else:
                        log(ERROR, f"CRITICAL: Key {key} missing from global parameters during client {client.cid} processing.")
                        valid_structure = False
                        break

            if valid_structure:
                weights_results_full_structure.append((client_full_ndarrays, fit_res.num_examples))
                total_examples += fit_res.num_examples
            else:
                log(WARNING, f"Skipping client {client.cid} due to parameter structure issues.")

        if not weights_results_full_structure:
             log(WARNING, "aggregate_fit: No valid client results to aggregate weights.")
             metrics_aggregated = self._aggregate_metrics(results, server_round) # Still aggregate metrics
             return self._current_parameters, metrics_aggregated # Return unchanged parameters

        # Aggregate weights (excluding head)
        aggregated_ndarrays_ordered = aggregate(weights_results_full_structure)

        final_aggregated_ndarrays_list = []
        aggregated_named_ndarrays = OrderedDict(zip(self._parameter_keys, aggregated_ndarrays_ordered))

        for key in self._parameter_keys:
            if key in aggregated_named_ndarrays:
                final_aggregated_ndarrays_list.append(aggregated_named_ndarrays[key])
            else:
                log(ERROR, f"CRITICAL: Aggregated key {key} missing after aggregation. Check aggregation logic and parameter keys.")
                if key in global_named_parameters:
                    log(WARNING, f"Using previous global value for missing aggregated key {key}.")
                    final_aggregated_ndarrays_list.append(global_named_parameters[key])
                else:
                    raise ValueError(f"Cannot find key {key} in aggregated results or previous global parameters.")

        parameters_aggregated = ndarrays_to_parameters(final_aggregated_ndarrays_list)
        metrics_aggregated = self._aggregate_metrics(results, server_round)

        log(INFO, f"aggregate_fit: aggregation complete for round {server_round}.")
        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results using weighted average loss and configured metrics function."""
        if not results:
            log(WARNING, "aggregate_evaluate: no results received.")
            return None, {}
        if not self.accept_failures and failures:
            log(WARNING, "aggregate_evaluate: failures received and accept_failures is False.")
            return None, {}

        # Filter results to only include successful ones
        successful_results = [(client, res) for client, res in results if res.status.code == Code.OK]
        if not successful_results:
            log(WARNING, "aggregate_evaluate: No successful evaluation results received.")
            return None, {}

        # Aggregate loss using weighted average logic from Flower
        loss_aggregated = weighted_loss_avg([(res.num_examples, res.loss) for _, res in successful_results if res.loss is not None])

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in successful_results if res.metrics]
            if eval_metrics:
                 metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            elif server_round == 1:
                 log(WARNING, "aggregate_evaluate: No metrics found in successful results to aggregate.")

        elif server_round == 1:
            log(WARNING, "aggregate_evaluate: No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, FitRes]], server_round: int) -> Dict[str, Scalar]:
        """Helper to aggregate metrics using fit_metrics_aggregation_fn, excluding modality and param_keys."""
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            excluded_keys = {"param_keys", "modality"}
            # Filter results: successful, have metrics, and prepare list for aggregation fn
            valid_metrics_results = [
                (res.num_examples, {k: v for k, v in res.metrics.items() if k not in excluded_keys})
                for _, res in results if res.status.code == Code.OK and res.metrics
            ]
            # Filter out results where the metrics dict became empty after exclusion
            fit_metrics = [(num, metrics) for num, metrics in valid_metrics_results if metrics]

            if fit_metrics:
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        elif server_round > 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return metrics_aggregated
