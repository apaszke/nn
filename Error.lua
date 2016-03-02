local Error = torch.class('nn.Error')


function Error:__init(msg)
    self.message = msg
    -- A network's "call stack". Container modules will handle the error and add
    -- themselves here, forming a reversed path from where error happened to the
    -- network topmost module.
    self.stack = {}
end

function Error:pushModule(module, i)
    table.insert(self.stack, {
        module = torch.type(module),
        index = i
    })
end

function Error:__tostring()
    local msg = "\n" -- Lua will automatically append some stuff - let's start from a new line
    for i = #self.stack,1,-1 do
        local info = self.stack[i]
        msg = msg .. string.format("In %s (%d):\n", info.module, info.index)
    end
    msg = msg .. self.message .. "\n"
    -- Unfortunately the final error will be thrown from a topmost container module
    msg = msg .. "\nWARNING: If you see a stack trace below, it doesn't point to the place where this error occured."
    return msg
end
