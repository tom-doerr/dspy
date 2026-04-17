import logging
import random

import numpy as np

import dspy
from dspy.teleprompt.simba import SIMBA
from dspy.teleprompt.simba_utils import prepare_models_for_resampling, wrap_program
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger("dspy.simbat")


class SIMBAT(SIMBA):
    """
    SIMBAT: SIMBA with Tail evaluation.
    
    A variant of SIMBA that evaluates on a "tail" subset of the training data
    instead of the full trainset during the final validation step.
    
    This is useful when you want to avoid evaluating on examples that were
    already seen during the minibatch optimization process.
    """
    
    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: list[dspy.Example],
        seed: int = 0,
        tail_eval_n: int = 256,
        dedup_seen: bool = True
    ):
        """
        Compile the student module using SIMBA optimization with tail evaluation.
        
        Args:
            student (dspy.Module): The module to optimize.
            trainset (list[dspy.Example]): The training dataset.
            seed (int): Random seed for reproducibility.
            tail_eval_n (int): Size of the tail evaluation window. Defaults to 256.
            dedup_seen (bool): If True, skip examples seen in minibatches during
                tail evaluation. Defaults to True.
        
        Returns:
            dspy.Module: The optimized module with the best performance on the tail set.
        """
        assert len(trainset) >= self.bsize, f"Trainset too small: {len(trainset)} < {self.bsize}"

        rng = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        programs, program_scores, next_program_idx = [], {}, 0
        
        def calc_average_score(prog_idx: int) -> float:
            scores = program_scores.get(prog_idx, [])
            return (sum(scores) / len(scores)) if scores else 0.0

        def top_k_plus_baseline(k: int) -> list[int]:
            scored = sorted(programs, key=lambda p: calc_average_score(p.simba_idx), reverse=True)
            top_k = [p.simba_idx for p in scored[:k]]
            if 0 not in top_k and len(top_k) > 0:
                top_k[-1] = 0
            return list(dict.fromkeys(top_k))

        def softmax_sample(rng_obj: random.Random, program_idxs: list[int], temperature: float) -> int:
            if not program_idxs:
                raise ValueError("No programs available for softmax sampling.")
            scores = [calc_average_score(idx) for idx in program_idxs]
            exps = [np.exp(s / temperature) for s in scores]
            sum_exps = sum(exps)
            if sum_exps <= 0:
                return rng_obj.choice(program_idxs)
            probs = [val / sum_exps for val in exps]
            return rng_obj.choices(program_idxs, weights=probs, k=1)[0]

        def register_new_program(prog: dspy.Module, score_list: list[float]):
            nonlocal next_program_idx
            next_program_idx += 1
            prog.simba_idx = next_program_idx
            programs.append(prog)
            program_scores[next_program_idx] = score_list

        # Baseline program
        student = student.deepcopy()
        student.simba_idx = 0
        programs.append(student)
        program_scores[0] = []
        winning_programs = [student]

        # Shuffled traversal state
        data_indices = list(range(len(trainset)))
        rng.shuffle(data_indices)
        instance_idx = 0
        seen_indices = []  # track all minibatch indices (for dedup)

        run_parallel = dspy.Parallel(access_examples=False, num_threads=self.num_threads)
        trial_logs = {}

        for batch_idx in range(self.max_steps):
            trial_logs[batch_idx] = {}
            logger.info(f"Starting batch {batch_idx+1} of {self.max_steps}.")

            # STEP 1: next batch (wrap + reshuffle on overflow)
            if instance_idx + self.bsize > len(trainset):
                rng.shuffle(data_indices)
                instance_idx = 0
            batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
            seen_indices.extend(batch_indices)
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize
            unique_seen_count = len(set(seen_indices))
            examples_seen_total = len(seen_indices)
            self._emit_progress(
                type="step_start",
                step_idx=batch_idx + 1,
                total_steps=self.max_steps,
                batch_size=self.bsize,
                batch_indices=batch_indices,
                trainset_size=len(trainset),
                examples_seen_total=examples_seen_total,
                unique_seen_count=unique_seen_count,
                sample_epoch=examples_seen_total / len(trainset),
                coverage_ratio=unique_seen_count / len(trainset),
            )

            # STEP 2: sample trajectories
            models = prepare_models_for_resampling(programs[0], self.num_candidates)
            top_programs = top_k_plus_baseline(self.num_candidates)

            exec_pairs, predictor2name = [], {}
            for model in models:
                for example in batch:
                    chosen_idx = softmax_sample(rng, top_programs, self.temperature_for_sampling)
                    candidate_system = programs[chosen_idx].deepcopy()
                    candidate_system.set_lm(model)
                    for name, predictor in candidate_system.named_predictors():
                        predictor2name[id(predictor)] = name
                    exec_pairs.append((wrap_program(candidate_system, self.metric), example))

            logger.info(f"Sampling program trajectories on {self.bsize} examples x {self.num_candidates} samples.")
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == self.bsize * self.num_candidates
            baseline_score = (
                sum(float(o["score"]) for o in outputs) / len(outputs)
                if outputs else 0.0
            )

            # STEP 3: bucket by example
            buckets = []
            batch_10p = np.percentile([float(o["score"]) for o in outputs], 10)
            batch_90p = np.percentile([float(o["score"]) for o in outputs], 90)
            for idx, _ in enumerate(batch):
                bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
                bucket.sort(key=lambda x: x["score"], reverse=True)
                max_score = float(bucket[0]["score"])
                min_score = float(bucket[-1]["score"])
                avg_score = sum(x["score"] for x in bucket) / len(bucket)
                buckets.append((bucket, (max_score - min_score, max_score, max_score - avg_score)))
            buckets.sort(key=lambda x: x[1], reverse=True)

            # STEP 4: build candidates by strategies (parallelized)
            # Phase 1: Prepare all candidates upfront
            candidates_to_process = []
            for bucket_idx, (bucket, _) in enumerate(buckets):
                src_prog_idx = softmax_sample(rng, top_k_plus_baseline(self.num_candidates), self.temperature_for_candidates)
                system_candidate = programs[src_prog_idx].deepcopy()

                # drop demos
                name2pred, num_demos_list = {}, []
                max_demos_tmp = self.max_demos if self.max_demos > 0 else 3
                for name, predictor in system_candidate.named_predictors():
                    name2pred[name] = predictor
                    num_demos_list.append(len(predictor.demos))
                num_demos = max(num_demos_list) if num_demos_list else 0
                num_to_drop = max(rng_np.poisson(num_demos / max_demos_tmp), int(num_demos >= max_demos_tmp))
                num_to_drop = min(num_to_drop, num_demos)
                drop_idx = {rng.randrange(num_demos) for _ in range(num_to_drop)}
                for _, predictor in name2pred.items():
                    predictor.demos = [demo for j, demo in enumerate(predictor.demos) if j not in drop_idx]

                strategy = rng.choice(self.strategies)
                logger.info(f"Batch {batch_idx+1}: Invoking strategy: {strategy.__name__}")
                candidates_to_process.append({
                    "bucket": bucket, "system_candidate": system_candidate,
                    "strategy": strategy, "predictor2name": predictor2name,
                    "name2predictor": name2pred, "prompt_model": None,
                    "batch_10p_score": batch_10p, "batch_90p_score": batch_90p,
                })

                if len(candidates_to_process) >= self.num_candidates + 1:
                    break

            def apply_strategy(item):
                try:
                    item["strategy"](item["bucket"], item["system_candidate"],
                        **{k: item[k] for k in ["predictor2name", "name2predictor",
                           "prompt_model", "batch_10p_score", "batch_90p_score"]})
                    return item["system_candidate"]
                except Exception as e:
                    logger.error(f"Strategy failed: {e}")

            strat_exec = ParallelExecutor(num_threads=self.num_threads, disable_progress_bar=True)
            results = strat_exec.execute(apply_strategy, candidates_to_process)
            system_candidates = [r for r in results if r is not None]

            # STEP 5–6: evaluate candidates on minibatch and average
            logger.info(f"Batch {batch_idx+1}: Evaluating {len(system_candidates)} programs on {self.bsize} examples.")
            exec_pairs = [(wrap_program(sys, self.metric), ex) for sys in system_candidates for ex in batch]
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == len(system_candidates) * self.bsize
            candidate_scores = []
            for i in range(len(system_candidates)):
                start, end = i * self.bsize, (i + 1) * self.bsize
                avg = sum(outputs[j]["score"] for j in range(start, end)) / self.bsize
                candidate_scores.append(avg)

            logger.info(
                f"Scores after {batch_idx+1} batches: {candidate_scores}, "
                f"Best: {max(candidate_scores) if candidate_scores else 'N/A'}\n"
            )

            # STEP 7: record best of this batch
            if candidate_scores:
                best_i = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
                winning_programs.append(system_candidates[best_i].deepcopy())

            # STEP 8: register all into global pool
            for i, cand in enumerate(system_candidates):
                start, end = i * self.bsize, (i + 1) * self.bsize
                register_new_program(cand, [outputs[j]["score"] for j in range(start, end)])

            self._emit_progress(
                type="step_end",
                step_idx=batch_idx + 1,
                total_steps=self.max_steps,
                batch_size=self.bsize,
                trainset_size=len(trainset),
                examples_seen_total=examples_seen_total,
                unique_seen_count=unique_seen_count,
                sample_epoch=examples_seen_total / len(trainset),
                coverage_ratio=unique_seen_count / len(trainset),
                baseline_score=baseline_score,
                candidate_count=len(system_candidates),
                best_candidate_score=max(candidate_scores) if candidate_scores else None,
            )

        # ---- CHANGED: validation on tail, not full trainset ----
        M = len(winning_programs) - 1
        N = self.num_candidates + 1
        program_idxs = ([0] * N) if M < 1 else [round(i * M / (N - 1)) for i in range(N)]
        program_idxs = list(dict.fromkeys(program_idxs))
        candidate_programs = [winning_programs[i].deepcopy() for i in program_idxs]

        # Build tail indices after last consumed minibatch window
        n = min(tail_eval_n, len(trainset))
        seen = set(seen_indices) if dedup_seen else set()

        # Handle small trainsets that cycled completely during training
        all_seen = dedup_seen and len(seen) >= len(trainset)
        if all_seen:
            logger.warning(f"All {len(trainset)} examples seen. Using random sample for tail eval.")
            eval_indices = rng.sample(range(len(trainset)), n)
        else:
            eval_indices, ptr = [], instance_idx
            while len(eval_indices) < n and len(seen) < len(trainset):
                if ptr >= len(data_indices):
                    ptr = 0
                idx = data_indices[ptr]
                ptr += 1
                if idx not in seen:
                    eval_indices.append(idx)
                    seen.add(idx)
        evalset = [trainset[i] for i in eval_indices]

        self._emit_progress(
            type="validation_start",
            total_steps=self.max_steps,
            trainset_size=len(trainset),
            candidate_count=len(candidate_programs),
            validation_size=len(evalset),
            validation_kind="tail",
        )
        logger.info(f"VALIDATION: Evaluating {len(candidate_programs)} programs on a {len(evalset)}-example tail set.")
        exec_pairs = [(wrap_program(sys, self.metric), ex) for sys in candidate_programs for ex in evalset]
        outputs = run_parallel(exec_pairs)

        scores = []
        for p_i in range(len(candidate_programs)):
            start, end = p_i * len(evalset), (p_i + 1) * len(evalset)
            sys_scores = [outputs[j]["score"] for j in range(start, end)]
            avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
            scores.append(avg_score)
            if p_i != 0:
                trial_logs[p_i - 1]["tail_score"] = avg_score

        best_idx = scores.index(max(scores)) if scores else 0
        best_program = candidate_programs[best_idx].deepcopy()
        best_program.candidate_programs = [{"score": s, "program": p} for s, p in zip(scores, candidate_programs, strict=False)]
        best_program.trial_logs = trial_logs
        
        logger.info(
            f"Final tail set scores: {scores}, Best: {max(scores) if scores else 'N/A'} "
            f"(at index {best_idx if scores else 'N/A'})\n\n\n"
        )
        self._emit_progress(
            type="validation_end",
            total_steps=self.max_steps,
            trainset_size=len(trainset),
            candidate_count=len(candidate_programs),
            validation_size=len(evalset),
            validation_kind="tail",
            validation_scores=scores,
            best_validation_score=max(scores) if scores else None,
            best_validation_index=best_idx if scores else None,
        )
        
        return best_program
