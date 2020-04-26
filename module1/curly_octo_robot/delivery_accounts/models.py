from django.db import models

from django.contrib.auth.models import BaseUserManager, AbstractBaseUser

class CORAccountManager(BaseUserManager):
    def create_user(self, email, username, password=None):
        if not email:
            return ValueError("Service providers must have email address")
        if not username:
            return ValueError("Service providers must have email address")
        user = self.model(
            email = self.normalize_email(email),
            username=username
        )
        user.set_password(password)
        user.save(user=self._db)
        return user

    def create_superuser(self, email, username, password):
        user = self.model(
            email = self.normalize_email(email),
            username=username
        )
        user.is_admin = True
        user.set_password(password)
        user.save(using=self._db)
        return user

class Account(AbstractBaseUser):
    email = models.EmailField(
        verbose_name='email',
        max_length=60,
        unique=True
    )
    username = models.CharField(
        max_length=30,
        unique=True
    )
    is_admin = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    servicetype = models.CharField(max_length=60)
    

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    objects = CORAccountManager()
    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        return self.is_admin
    
    def has_module_perms(self, app_label):
        return True