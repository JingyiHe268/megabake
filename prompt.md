See ARCHITECTURE.md and north-star.md and try to understand the main crux of this project and lets try to 
do things from first principles here. if you see the main branch or any other branch except v4-impl you will see a submoptimal bad compiler
that is not good for an actual megakernel usecase. the main problem in the version in branch (I think the best yet) is that even that 
suffers from not having good enough gemms or basically cublass level kernels to stitch since cublass can't be stitched in 
megakernels. I am working on it but the kernels that I am stitching are always losing to cublass level kernels and 
eventually this failure catches up and any decent size models dont get any benefits infact they almost always lose to torch.compile.
I have tried to make a good enough architecture in ARCHITECTURE.md but I dont want to do it completely again and in the end find out that 
this arch is also shit. 
I have stumbled upon several things and hence svseral direction in my research while looking at things here and there
One of them is actual codebases of mirage MPK compiler. its not what we want but there are a lot of ideas and infact implementation that I think we can straight up copy from them like things like get_dtensor_tile_layout and dimesnions and scheduling etc maybe we can copy the hard parts of scheduling and figuring out tiling infra from them maybe. the codebase is cloned in agent_space also if you wanna take a look.
secondly I think prefill and ddecode kernels should be different but the same compiler should handle that made just a flag called `--mode prefill/decode` should work and it should chose ideal scheduling criteria like tile, block, cta, mma layout etc etc for prefill vs dcdecode. I have seen this in tpu megakernels in inferact for this whose code is also cloned in agent_space. 
I also stumbled upon this https://accu.org/journals/overload/32/181/schuetze/ and it is giving me a very strong itch to pursue this direction as well but idk if it will be wirthwhile or not to get reconstructing cublass lebvel kernels by reverse engineering them  or not or will it be a time-waste but gives me
very string intutions and itches about the scheduling parts of our compiler. whatever I have till now is just my research and doesn't mean it is the best version of things. I am just exploring things at my level right now. my IRs and compiler design and IR passes might be good might be shit idk. I want you to research in book like book.mlc.ai and ither ml compiler design or simply otehr actual compilers. I also https://github.com/pytorch/FBGEMM and idk
how helpful this will actually be or not we might consider this also if it suffices our use case or even this idk https://github.com/deepseek-ai/DeepGEMM
Re-architect if you want to or have to but I want a final good compile design for our use case. the only thing you are not allowed to change is the scope of the project which is essentially north-star.md 
I want the response and design to be made followed and thought through the lense of absolute first principles. I believe that beauty is in simplicity 
and simple well understandable systems are better and performant than un-necessarily complex ones. I also want you to follow 80% of the way till ASD-STE100.
while giving the final output for new md files or edit the current md files. 
These are my research directions and things but you can have your own thing as well and research on that as well. its upto you.
you are free to run experiments also if you want to before making decisions. I would want you to make informed decisions. 