-- This is code common to container modules, which are collections of
-- smaller constituent modules like Parallel, Sequential, etc.
local Container, parent = torch.class('nn.Container', 'nn.Module')

function Container:__init(...)
    parent.__init(self, ...)
    self.modules = {}
end

function Container:add(module)
    table.insert(self.modules, module)
    return self
end

function Container:get(index)
    return self.modules[index]
end

function Container:size()
    return #self.modules
end

-- module argument can be retrieved with moduleIndex, but code is cleaner when
-- it has to be specified anyway.
function Container:rethrowErrors(module, moduleIndex, funcName, ...)
    local ERROR_HEADER = "======== ERROR ========"
    local TRACEBACK_WARNING = "WARNING: If you see a stack trace below, it doesn't point to the place where this error occured. Please use only the one above."
    local function handleError(err)
       local err_lines = err:split('\n')
       -- This should be executed only in the first container that handles the
       -- error. First line will be appended by lua when calling error, then
       -- there should be our header.
       if #err_lines < 2 or err_lines[2] ~= ERROR_HEADER then
          local traceback = debug.traceback()
          -- Remove this handler from the stack
          local _, first_line_len = traceback:find('^.-\n')
          local _, second_line_end = traceback:find('^.-\n.-\n')
          traceback = traceback:sub(1, first_line_len) .. traceback:sub(second_line_end+1)
          err = err .. '\n' .. traceback .. '\n\n' .. TRACEBACK_WARNING
       else
          -- +2 for newlines
          local second_line_end = #err_lines[1] + #err_lines[2] + 2
          err = err:sub(second_line_end+1)
       end
       local msg = string.format('In %d module of %s',
                                 moduleIndex, torch.type(self))
       -- Preceding newline has to be here, because lua will prepend a file path.
       err = '\n' .. ERROR_HEADER .. '\n' .. msg .. '\n' .. err
       return err
    end

    assert(module == self.modules[moduleIndex], "mismatch between moduleIndex and self.modules in rethrowErrors")
    local args = {...}
    local ok, ret, noret = xpcall(function()
                                     return module[funcName](module, unpack(args))
                                  end,
                                  handleError)
    assert(noret == nil, "rethrowErrors supports only one return argument")
    if not ok then
       error(ret)
    end
    return ret
end

function Container:applyToModules(func)
    for _, module in ipairs(self.modules) do
        func(module)
    end
end

function Container:zeroGradParameters()
    self:applyToModules(function(module) module:zeroGradParameters() end)
end

function Container:updateParameters(learningRate)
    self:applyToModules(function(module) module:updateParameters(learningRate) end)
end

function Container:training()
    self:applyToModules(function(module) module:training() end)
    parent.training(self)
end

function Container:evaluate()
    self:applyToModules(function(module) module:evaluate() end)
    parent.evaluate(self)
end

function Container:share(mlp, ...)
    for i=1,#self.modules do
        self.modules[i]:share(mlp.modules[i], ...);
    end
end

function Container:reset(stdv)
    self:applyToModules(function(module) module:reset(stdv) end)
end

function Container:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}
    for i=1,#self.modules do
        local mw,mgw = self.modules[i]:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    return w,gw
end

function Container:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   if self.modules then
      for i,module in pairs(self.modules) do
         module:clearState()
      end
   end
   return self
end
