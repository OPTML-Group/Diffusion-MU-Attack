
class Attacker:
    def __init__(
                self,
                iteration,
                seed_iteration,
                insertion_location,
                k,
                eval_seed,
                attack_idx,
                universal,
                sequential,
                ):
        self.iteration = iteration
        self.seed_iteration = seed_iteration
        self.insertion_location = insertion_location
        self.k = k
        self.eval_seed = eval_seed
        self.universal = universal
        self.attack_idx = attack_idx
        self.sequential = sequential
        