we have a lot of redudant types for example StepResult and PipelineStepResult are the same 

loop.py should also be using the models defined in loop_types that reimplement the stubs 

this is also a bigger refactor but we need to start actually using the langfuse Evaluation api, it has an api for visualizing scores. memoizing the baseline jc test is a good idea as we are already doing. 

we also need to consolidate the prompt into a single templated file. 

according to the tracing its now off too, but that will fall out of the refactor to an actual repoqa type benchmark 

we should prob include a script in here to easily spit out state machine visualizations, its lowkey kinda a mess to read

we should split more conceptually into different better named files/folders now that we have a strong model of the two loops
