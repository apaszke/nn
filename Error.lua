local Error = torch.class('nn.Error')

function Error:__init(msg, traceback)
   self.message = msg
   self.traceback = traceback
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

function Error:ordinalSuffix(n)
   if n >= 11 and n <= 13 then
      return "th"
   elseif n % 10 == 1 then
      return "st"
   elseif n % 10 == 2 then
      return "nd"
   elseif n % 10 == 3 then
      return "rd"
   end

   return "th"
end

function Error:__tostring()
   local msg = "\n" -- Lua will automatically append some stuff - let's start from a new line
   for i = #self.stack,1,-1 do
      local info = self.stack[i]
      msg = msg .. string.format("In %d%s module of %s:\n", info.index,
                                 self:ordinalSuffix(info.index), info.module)
   end
   msg = msg .. self.traceback .. "\n\n"
   msg = msg .. "WARNING: If you see a stack trace below, it doesn't point to the place where this error occured. Please use only the one above.\n"
   msg = msg .. self.message
   return msg
end
