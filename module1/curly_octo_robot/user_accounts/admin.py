from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from user_accounts.models import Account

class AccountAdmin(UserAdmin):
    list_display = ('email','username','latitude','longitude')
    search_fields = ('email','username')
    readonly_fields = ()
    filter_horizontal = ()
    list_filter = ()
    fieldsets = ()

admin.site.register(Account, AccountAdmin)